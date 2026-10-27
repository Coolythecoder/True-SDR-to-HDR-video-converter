"""Nonblocking OpenCL preview selection with per-result CPU validation.

A valid (especially black) first frame cannot authorize later frames, grades or
sizes. Never publish unchecked GPU pixels. This safety check costs a CPU preview
render; it does not change export acceleration or the YouTube encoding recipe.
"""
import numpy as np
from creative_hdr import active_grade
from gpu_colour import GPUPlan, GPU_DEVICES
from grading_preview import render_command


class PreviewAcceleration:
    def __init__(self, engine):
        self.engine = engine
        self.enabled = True
        self.device_index = 0
        # Status of the most recently accepted result, not a permanent bypass.
        self.verified = False
        self.plans = set()
        self.backend = 'CPU preview'
        self._generation = 0
        self._fallback_reason = ''

    def reset(self, enabled):
        self._generation += 1
        self.enabled = bool(enabled)
        self.device_index = 0
        self.verified = False
        self._fallback_reason = ''
        self.backend = 'CPU preview'

    def close(self):
        self._generation += 1
        for plan in tuple(self.plans):
            plan.close()
        self.plans.clear()

    def render(self, width, height, grade, white, callback, payload, limit):
        engine = self.engine
        if not self.enabled or not active_grade(grade):
            self.backend = 'CPU preview'
            return False
        if self.device_index >= len(GPU_DEVICES):
            self.backend = self._fallback_reason or 'CPU preview (GPU unavailable)'
            return False
        plan = GPUPlan(grade, white, GPU_DEVICES[self.device_index])
        self.plans.add(plan)
        revision = engine.revision
        key = engine._key()
        source = engine.source
        ffmpeg = engine.ffmpeg
        generation = self._generation
        self.verified = False

        def cleanup():
            plan.close()
            self.plans.discard(plan)

        def current():
            return (not engine.closed and not engine.paused and self.enabled
                    and generation == self._generation and engine.source is source
                    and engine.source is not None and engine.ffmpeg == ffmpeg
                    and revision == engine.revision and key == engine._key())

        def reschedule():
            if not engine.closed and not engine.paused and engine.source is not None:
                engine.debounce.start()

        def fallback(message):
            if not current():
                reschedule()
                return
            self.device_index += 1
            self.verified = False
            self._fallback_reason = 'CPU preview (GPU fallback)'
            self.backend = self._fallback_reason
            detail = str(message).replace('\n', ' ')[-400:]
            engine.message.emit('GPU preview unavailable; trying a compatible backend / CPU. ' + detail)
            reschedule()

        def reject_backend(reason):
            self.device_index = len(GPU_DEVICES)
            self.verified = False
            self._fallback_reason = reason
            self.backend = reason

        def reference_failed(message):
            if not current():
                reschedule()
                return
            reject_backend('CPU preview (GPU reference validation failed)')
            engine.failed.emit('Cannot validate this preview; GPU pixels were not displayed. ' +
                               str(message)[-400:])

        def gpu_ready(data):
            if not current():
                reschedule()
                return
            if len(data) != width*height*6:
                fallback('Incomplete GPU preview.')
                return
            # Validate EVERY requested result against the same cached float
            # frame and settings. A previous success says nothing about a new
            # kernel, geometry, source, or runtime/driver failure. Do not show
            # the GPU result even briefly while this check is outstanding.
            def reference_ready(reference):
                if not current():
                    reschedule()
                    return
                if len(reference) != width*height*6:
                    reference_failed('Incomplete CPU comparison.')
                    return
                diff = np.abs(np.frombuffer(reference, dtype=np.uint8).astype(np.int16)
                              - np.frombuffer(data, dtype=np.uint8).astype(np.int16))
                maximum, average = int(diff.max()), float(diff.mean())
                if maximum > 2 or average > .15:
                    # Remain on CPU until acceleration is explicitly reset.
                    # This is preview-local: do not modify export settings.
                    reject_backend('CPU preview (GPU parity check failed)')
                    engine.message.emit(
                        'GPU preview rejected (max difference {}, mean {:.3f}); '
                        'using the complete CPU preview. Export settings are unchanged.'.format(
                            maximum, average))
                    callback(reference)
                    return
                self.verified = True
                self.backend = 'OpenCL GPU grading'
                callback(data)
            engine._launch(render_command(ffmpeg, width, height, grade, white),
                           'render', reference_ready, payload=payload, limit=limit,
                           on_error=reference_failed)
        engine._launch(render_command(ffmpeg, width, height, grade, white, gpu_plan=plan),
                       'render', gpu_ready, payload=payload, limit=limit,
                       on_error=fallback, cleanup=cleanup)
        return True
