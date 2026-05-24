# services/barge_in_engine.py
import asyncio
import logging

logger = logging.getLogger("barge_in_engine")
logger.setLevel(logging.INFO)

class InterruptionCapablePlaybackQueue:
    def __init__(self):
        self.playback_queue = asyncio.Queue()
        self.current_playback_task = None
        self.last_interrupt_snapshot = None

    async def enqueue_audio_chunk(self, audio_chunk: bytes):
        """
        Pushes synthesized response frames from Kami's active persona 
        directly into the speaker outbound execution track.
        """
        await self.playback_queue.put(audio_chunk)

    async def start_speaker_delivery_loop(self):
        """
        Simulates flushing PCM audio buffers down LiveKit's outbound WebRTC track.
        """
        try:
            while True:
                chunk = await self.playback_queue.get()
                
                # Wrap the direct hardware streaming block in an interruptible execution context
                self.current_playback_task = asyncio.create_task(self._render_chunk_to_hardware(chunk))
                try:
                    await self.current_playback_task
                except asyncio.CancelledError:
                    logger.warning("[BARGE-IN] Active voice synthesis playback was explicitly aborted mid-frame!")
                finally:
                    self.playback_queue.task_done()
                    self.current_playback_task = None
                    
        except asyncio.CancelledError:
            logger.info("Outbound audio delivery track smoothly unmounted.")

    async def _render_chunk_to_hardware(self, chunk: bytes):
        # Simulates 400ms duration of audio playback delivery
        await asyncio.sleep(0.4)

    def handle_user_barge_in_event(self):
        """
        The moment LiveKit registers an VAD (Voice Activity Detection) flag indicating the 
        learner began speaking, this triggers. Drops upcoming responses immediately.
        """
        logger.info("[BARGE-IN ENGAGED] User interruption registered. Purging outbound audio pipeline.")
        queued_before = self.playback_queue.qsize()
        active_cancelled = False
        
        # 1. Kill active rendering immediately
        if self.current_playback_task and not self.current_playback_task.done():
            self.current_playback_task.cancel()
            active_cancelled = True
            
        # 2. Flush any accumulated unplayed audio sentences completely out of memory
        purged_chunks = 0
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
                self.playback_queue.task_done()
                purged_chunks += 1
            except asyncio.QueueEmpty:
                break
                
        self.last_interrupt_snapshot = {
            "queued_before": queued_before,
            "purged_chunks": purged_chunks,
            "queued_after": self.playback_queue.qsize(),
            "active_task_cancelled": active_cancelled,
        }
        logger.info("[BARGE-IN COMPLETE] Speaker queues are clean. Ready for inbound speech capture.")
        return self.last_interrupt_snapshot
