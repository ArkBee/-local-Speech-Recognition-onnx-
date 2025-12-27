import asyncio
import sounddevice as sd
import numpy as np
import logging
from collections import deque
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class AsyncAudioRecorder:
    """
    Асинхронный рекордер аудио с использованием sounddevice.
    Позволяет начинать и останавливать запись, собирая аудиоданные в буфер.
    """
    DEFAULT_SAMPLERATE = 48000  # Гц, стандартная для многих моделей распознавания
    DEFAULT_CHANNELS = 1        # Моно
    DEFAULT_DTYPE = 'int16'     # Тип данных для аудио
    # Размер чанка в кадрах. Меньше чанк -> чаще callback, но меньше задержка.
    # Больше чанк -> реже callback, но может быть больше задержка при старте/стопе.
    # 1024 кадра при 48000 Гц = ~21.3 мс.
    DEFAULT_BLOCKSIZE = 1024 

    def __init__(self, 
                 samplerate: int = DEFAULT_SAMPLERATE, 
                 channels: int = DEFAULT_CHANNELS,
                 dtype: str = DEFAULT_DTYPE,
                 blocksize: int = DEFAULT_BLOCKSIZE):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize # Размер блока для InputStream

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream: Optional[sd.InputStream] = None
        self._audio_buffer: deque[np.ndarray] = deque()
        self._is_recording = False
        self._recording_task: Optional[asyncio.Task] = None
        
        # Событие для сигнализации callback'у о необходимости прекратить добавлять данные
        self._stop_event: Optional[asyncio.Event] = None 

        # Проверка доступности микрофона при инициализации (опционально)
        try:
            sd.check_input_settings(samplerate=self.samplerate, channels=self.channels, dtype=self.dtype)
            logger.info(f"Audio input device available with settings: SR={samplerate}, CH={channels}, DT={dtype}")
        except Exception as e:
            logger.error(f"Failed to validate audio input settings: {e}", exc_info=True)
            # Можно здесь выбросить исключение, чтобы приложение не стартовало без микрофона
            # raise RuntimeError(f"Audio input device not properly configured or available: {e}") from e


    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        """
        Callback-функция, вызываемая sounddevice с новыми аудиоданными.
        Выполняется в отдельном потоке sounddevice.
        """
        if status:
            logger.warning(f"Sounddevice callback status: {status}")
        
        if self._is_recording and self._stop_event and not self._stop_event.is_set():
            # Копируем данные, так как indata может быть перезаписан
            self._audio_buffer.append(indata.copy())
        # else:
            # logger.debug("Callback invoked but not recording or stop event set.")


    async def start_recording(self):
        """
        Начинает асинхронную запись аудио.
        """
        if self._is_recording:
            logger.warning("Recording is already in progress.")
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("Cannot start recording: asyncio event loop not found.")
            return

        logger.info(f"Starting audio recording... (SR={self.samplerate}, CH={self.channels})")
        self._audio_buffer.clear()
        self._is_recording = True
        self._stop_event = asyncio.Event() # Создаем новое событие для этой сессии записи

        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.blocksize, # Используем blocksize для управления частотой callback'ов
                callback=self._audio_callback
            )
            self._stream.start()
            logger.info("Audio stream started.")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}", exc_info=True)
            self._is_recording = False
            self._stream = None # Убедимся, что стрим сброшен
            self._stop_event = None
            return

        # Основной цикл записи не нужен, так как callback сам наполняет буфер.
        # Просто держим состояние _is_recording.

    async def stop_recording(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        Останавливает асинхронную запись аудио и возвращает записанные данные.

        Returns:
            Кортеж (аудиоданные_как_numpy_array, частота_дискретизации) или None, если ничего не записано.
        """
        if not self._is_recording:
            logger.warning("Recording is not in progress or already stopped.")
            if self._audio_buffer: # Если что-то осталось в буфере от предыдущей прерванной записи
                try:
                    recorded_data = np.concatenate(list(self._audio_buffer))
                    self._audio_buffer.clear()
                    logger.info(f"Returning residual audio data from buffer: {recorded_data.shape}")
                    return recorded_data, self.samplerate
                except ValueError: # Буфер пуст
                    self._audio_buffer.clear() # На всякий случай
                    return None
            return None

        logger.info("Stopping audio recording...")
        self._is_recording = False # Сначала меняем флаг

        if self._stop_event:
            self._stop_event.set() # Сигнализируем callback'у прекратить добавлять данные

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
                logger.info("Audio stream stopped and closed.")
            except Exception as e:
                logger.error(f"Error stopping/closing audio stream: {e}", exc_info=True)
            finally:
                self._stream = None
        
        # Небольшая задержка, чтобы дать последним callback'ам шанс обработаться,
        # особенно если _stop_event.set() произошло чуть позже фактической остановки стрима.
        # Это может быть не идеально, но помогает собрать все хвосты.
        await asyncio.sleep(0.05 * (self.blocksize / self.samplerate)) # Пауза, пропорциональная длине блока

        if not self._audio_buffer:
            logger.info("No audio data recorded.")
            return None

        try:
            # Собираем все чанки из deque
            # Важно: deque может быть модифицирован callback'ом, но после stream.stop() и stop_event.set()
            # новые данные добавляться не должны. Копируем в list перед concatenate.
            # logger.debug(f"Buffer size before concat: {len(self._audio_buffer)}")
            # for i, chunk in enumerate(list(self._audio_buffer)):
            #    logger.debug(f"Chunk {i} shape: {chunk.shape}, dtype: {chunk.dtype}")

            recorded_data = np.concatenate(list(self._audio_buffer))
            self._audio_buffer.clear() # Очищаем буфер после извлечения данных
            logger.info(f"Audio recording stopped. Total samples: {len(recorded_data)}")
            return recorded_data, self.samplerate
        except ValueError: # Если буфер оказался пуст к моменту конкатенации
            logger.info("Audio buffer was empty at concatenation, no data returned.")
            self._audio_buffer.clear() # Убедимся, что он пуст
            return None
        except Exception as e:
            logger.error(f"Error concatenating audio buffer: {e}", exc_info=True)
            self._audio_buffer.clear()
            return None

    def is_recording(self) -> bool:
        return self._is_recording

# Пример использования (для тестирования этого модуля отдельно)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def test_recorder():
        logger.info("Starting audio recorder test.")
        recorder = AsyncAudioRecorder(samplerate=16000) # Для теста можно понизить SR

        if not recorder._loop: # Если _loop не был установлен (например, из-за ошибки в __init__)
             try:
                recorder._loop = asyncio.get_running_loop()
             except RuntimeError:
                logger.error("Failed to get running loop for test.")
                return


        logger.info("Test 1: Short recording (3 seconds)")
        await recorder.start_recording()
        logger.info("Recording... Press Ctrl+C to stop early if needed.")
        await asyncio.sleep(3)
        audio_data_tuple = await recorder.stop_recording()

        if audio_data_tuple:
            audio, sr = audio_data_tuple
            logger.info(f"Test 1: Recorded {len(audio)} samples at {sr} Hz. Duration: {len(audio)/sr:.2f}s. Dtype: {audio.dtype}")
            # Здесь можно было бы сохранить в файл для проверки
            # import soundfile as sf
            # sf.write("test_recording_1.wav", audio, sr)
            # logger.info("Test 1: Saved to test_recording_1.wav")
        else:
            logger.warning("Test 1: No audio data returned.")

        await asyncio.sleep(1) # Пауза между тестами

        logger.info("Test 2: Multiple short recordings")
        for i in range(3):
            logger.info(f"Test 2, Iteration {i+1}: Starting recording for 1 second.")
            await recorder.start_recording()
            await asyncio.sleep(1)
            audio_data_tuple_multi = await recorder.stop_recording()
            if audio_data_tuple_multi:
                audio_multi, sr_multi = audio_data_tuple_multi
                logger.info(f"Test 2, Iteration {i+1}: Recorded {len(audio_multi)} samples at {sr_multi} Hz. Duration: {len(audio_multi)/sr_multi:.2f}s")
                # sf.write(f"test_multi_{i+1}.wav", audio_multi, sr_multi)
            else:
                logger.warning(f"Test 2, Iteration {i+1}: No audio data returned.")
            await asyncio.sleep(0.5) # Короткая пауза

        logger.info("Test 3: Start/Stop quickly")
        logger.info("Test 3: Starting recording...")
        await recorder.start_recording()
        await asyncio.sleep(0.1) # Очень короткая запись
        logger.info("Test 3: Stopping recording...")
        audio_data_tuple_quick = await recorder.stop_recording()
        if audio_data_tuple_quick:
            audio_q, sr_q = audio_data_tuple_quick
            logger.info(f"Test 3: Recorded {len(audio_q)} samples at {sr_q} Hz. Duration: {len(audio_q)/sr_q:.2f}s")
        else:
            logger.warning("Test 3: No audio data returned.")
            
        logger.info("Audio recorder test finished.")

    try:
        asyncio.run(test_recorder())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in test_recorder: {e}", exc_info=True)
