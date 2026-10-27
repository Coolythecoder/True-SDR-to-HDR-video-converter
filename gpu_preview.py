"""Nonblocking OpenCL preview selection and CPU fallback through QProcess."""
import numpy as np
from creative_hdr import active_grade
from gpu_colour import GPUPlan, GPU_DEVICES
from grading_preview import render_command


class PreviewAcceleration:
    def __init__(self, engine):
        self.engine = engine
        self.enabled = True
        self.device_index = 0
        self.verified = False
        self.plans = set()
        self.backend = 'CPU preview'

    def reset(self, enabled):
        self.enabled = bool(enabled)
        self.device_index = 0
        self.verified = False
        self.backend = 'CPU preview'

    def close(self):
        for plan in tuple(self.plans):
            plan.close()
        self.plans.clear()

    def render(self, width, height, grade, white, callback, payload, limit):
        engine = self.engine
        if not self.enabled or not active_grade(grade) or self.device_index >= len(GPU_DEVICES):
            self.backend = 'CPU preview'
            return False
        plan = GPUPlan(grade, white, GPU_DEVICES[self.device_index])
        self.plans.add(plan)
        revision = engine.revision
        key = engine._key()

        def cleanup():
            plan.close()
            self.plans.discard(plan)

        def current():
            return (not engine.closed and not engine.paused and engine.source is not None
                    and revision == engine.revision and key == engine._key())

        def fallback(message):
            if not current():
                if not engine.closed and not engine.paused:
                    engine.debounce.start()
                return
            self.device_index += 1
            self.verified = False
            self.backend = 'CPU preview (GPU fallback)'
            engine.message.emit('GPU preview unavailable; trying a compatible backend / CPU.')
            engine.debounce.start()

        def gpu_ready(data):
            if not current():
                engine.debounce.start()
                return
            if len(data) != width*height*6:
                fallback('Incomplete GPU preview.')
                return
            if self.verified:
                self.backend = 'OpenCL GPU grading'
                callback(data)
                return
            # Initial device selection is checked against the same CPU-rendered
            # frame. This remains asynchronous and is not an HDR display proof.
            def reference_ready(reference):
                if not current():
                    engine.debounce.start()
                    return
                if len(reference) != len(data):
                    fallback('Incomplete CPU comparison.')
                    return
                diff = np.abs(np.frombuffer(reference,dtype=np.uint8).astype(np.int16)
                              -np.frombuffer(data,dtype=np.uint8).astype(np.int16))
                if diff.max() > 2 or diff.mean() > .15:
                    self.device_index = len(GPU_DEVICES)
                    self.backend = 'CPU preview (GPU parity check failed)'
                    callback(reference)
                    return
                self.verified = True
                self.backend = 'OpenCL GPU grading'
                callback(data)
            engine._launch(render_command(engine.ffmpeg,width,height,grade,white),
                           'render',reference_ready,payload=payload,limit=limit)
        engine._launch(render_command(engine.ffmpeg,width,height,grade,white,gpu_plan=plan),
                       'render',gpu_ready,payload=payload,limit=limit,on_error=fallback,cleanup=cleanup)
        return True
