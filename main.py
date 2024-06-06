import argparse
import multiprocessing
import subprocess
from datetime import timedelta, datetime
from typing import Iterable, Tuple, Optional

import m3u8
import requests
import tempfile
import shutil
import os
from faster_whisper import WhisperModel
import torch
from faster_whisper.transcribe import Segment
import copy
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import threading
import time
from multiprocessing.pool import ThreadPool
from functools import partial
from faster_whisper.utils import available_models
from m3u8 import PlaylistList, SegmentList

translated_chunk_paths = {}
chunk_to_vtt = {}

CHUNK_LIST_BASE_URI = None
BASE_PLAYLIST_SER = None
CHUNK_LIST_SER = None
SUB_LIST_SER = None

TARGET_BUFFER_SECS = 60
MAX_TARGET_BUFFER_SECS = 120


def segments_to_srt(segments: Iterable[Segment], ts_offset: timedelta) -> str:
    base_ts = datetime(1970, 1, 1, 0, 0, 0) + ts_offset
    segment_chunks = [
        f'{i + 1}\n{(base_ts + timedelta(seconds=segment.start)).strftime('%H:%M:%S,%f')[:-3]} --> {(base_ts + timedelta(seconds=segment.end)).strftime('%H:%M:%S,%f')[:-3]}\n{segment.text}'
        for i, segment in enumerate(segments)]
    return '\n\n'.join(segment_chunks)


def segments_to_webvtt(segments: Iterable[Segment], ts_offset: timedelta) -> str:
    base_ts = datetime(1970, 1, 1, 0, 0, 0) + ts_offset
    segment_chunks = [
        f'{i + 1}\n{(base_ts + timedelta(seconds=segment.start)).strftime('%H:%M:%S.%f')[:-3]} --> {(base_ts + timedelta(seconds=segment.end)).strftime('%H:%M:%S.%f')[:-3]}\n{segment.text}'
        for i, segment in enumerate(segments)]
    return 'WEBVTT\n\n' + '\n\n'.join(segment_chunks)


def download_chunk_and_transcribe(session: requests.Session, model: WhisperModel, absolute_url: str, segment_uri: str,
                                  temp_chunk_dir: str, hard_subs: bool, translate: bool, beam_size: int,
                                  vad_filter: bool, language: Optional[str]) -> str:
    with tempfile.NamedTemporaryFile(dir=temp_chunk_dir, delete=False, delete_on_close=False,
                                     suffix='.ts') as chunk_fp:
        with session.get(absolute_url, stream=True) as r:
            shutil.copyfileobj(r.raw, chunk_fp)

        chunk_fp.close()

        start_ts = timedelta(seconds=float(subprocess.check_output(['ffprobe', '-i', chunk_fp.name, '-show_entries',
                                                                    'stream=start_time', '-loglevel', 'quiet',
                                                                    '-select_streams', 'a:0', '-of',
                                                                    'csv=p=0']).splitlines()[0]))

        segments, _ = model.transcribe(chunk_fp.name, beam_size=beam_size, vad_filter=vad_filter, language=language,
                                       task='translate' if translate else 'transcribe')

        if hard_subs:
            with tempfile.NamedTemporaryFile(dir=temp_chunk_dir, delete_on_close=False, suffix='.srt') as srt_file:
                srt_content = segments_to_srt(segments, start_ts)
                if not srt_content:
                    return chunk_fp.name

                srt_file.write(bytes(srt_content, 'utf-8'))

                srt_file.close()
                chunk_fp_name_split = os.path.splitext(chunk_fp.name)
                translated_chunk_name = chunk_fp_name_split[0] + '_trans' + chunk_fp_name_split[1]

                subprocess.check_output(['ffmpeg', '-hwaccel', 'auto', '-i', chunk_fp.name, '-copyts',
                                         '-muxpreload', '0', '-muxdelay', '0', '-preset', 'ultrafast', '-c:a', 'copy',
                                         '-loglevel', 'quiet', '-vf', f'subtitles={os.path.basename(srt_file.name)}',
                                         translated_chunk_name])

                os.unlink(chunk_fp.name)
                return translated_chunk_name
        else:
            vtt_uri = os.path.splitext(segment_uri)[0] + '.vtt'
            chunk_to_vtt[vtt_uri] = segments_to_webvtt(segments, start_ts)
            return chunk_fp.name


class HTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        response_content = None
        if self.path == '/playlist.m3u8':
            response_content = BASE_PLAYLIST_SER
        elif self.path == '/chunklist.m3u8':
            response_content = CHUNK_LIST_SER
        elif self.path == '/subs.m3u8':
            response_content = SUB_LIST_SER
        elif self.path in translated_chunk_paths:
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp2t')
            self.send_header('Content-Length', str(os.path.getsize(translated_chunk_paths[self.path])))
            self.end_headers()

            with open(translated_chunk_paths[self.path], 'rb') as f:
                shutil.copyfileobj(f, self.wfile)

            return
        elif self.path in chunk_to_vtt:
            response_content = bytes(chunk_to_vtt[self.path], 'utf-8')

        self.send_response(200 if response_content else 404)

        if self.path.endswith('.m3u8'):
            self.send_header('Content-Type', 'application/vnd.apple.mpegurl')
        elif self.path.endswith('.vtt'):
            self.send_header('Content-Type', 'text/vtt')

        if response_content:
            self.send_header('Content-Length', str(len(response_content)))

        self.end_headers()
        if response_content:
            self.wfile.write(response_content)


def http_listener(server_address: Tuple[str, int]):
    server = ThreadingHTTPServer(server_address, HTTPHandler)
    server.serve_forever()


def normalise_chunk_uri(chunk_uri: str) -> str:
    chunk_uri = os.path.splitext(chunk_uri)[0] + '.ts'
    chunk_uri = chunk_uri.replace('../', '').replace('./', '')
    return '/' + chunk_uri


def download_and_transcribe_wrapper(segment: Segment, session: requests.Session, model: WhisperModel, base_uri: str,
                                    chunk_dir: str, hard_subs: bool, translate: bool, beam_size: int,
                                    vad_filter: bool, language: Optional[str]):
    chunk_url = os.path.join(base_uri, segment.uri)
    chunk_uri = normalise_chunk_uri(segment.uri)
    if chunk_uri not in translated_chunk_paths:
        translated_chunk_paths[chunk_uri] = download_chunk_and_transcribe(session, model, chunk_url, chunk_uri,
                                                                          chunk_dir, hard_subs, translate,
                                                                          beam_size, vad_filter, language)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='livevtt')
    parser.add_argument('-u', '--url', required=True)
    parser.add_argument('-s', '--hard-subs', action='store_true',
                        help='Set if you want the subtitles to be baked into the stream itself')
    parser.add_argument('-l', '--bind-address', type=str, help='The IP address to bind to '
                                                               '(defaults to 127.0.0.1)', default='127.0.0.1')
    parser.add_argument('-p', '--bind-port', type=int, help='The port to bind to (defaults to 8000)',
                        default=8000)
    parser.add_argument('-m', '--model', type=str, help='Whisper model to use (defaults to large)',
                        default='large', choices=available_models())
    parser.add_argument('-b', '--beam-size', type=int, help='Beam size to use (defaults to 5)', default=5)
    parser.add_argument('-c', '--use-cuda', type=bool, help='Use CUDA where available. Defaults to true',
                        default=True)
    parser.add_argument('-t', '--transcribe', action='store_true',
                        help='If set, transcribes rather than translates the given stream.')
    parser.add_argument('-vf', '--vad-filter', type=bool, help='Whether to utilise the Silero VAD model '
                                                               'to try and filter out silences. Defaults to false.',
                        default=False)
    parser.add_argument('-la', '--language', type=str, help='The original language of the stream, '
                                                            'if known/not multi-lingual. Can be left unset.')
    parser.add_argument('-ua', '--user-agent', type=str, help='User agent to use to retrieve playlists / '
                                                              'stream chunks.', default='VLC/3.0.18 LibVLC/3.0.18')

    args = parser.parse_args()

    threading.Thread(target=http_listener, daemon=True, args=((args.bind_address, args.bind_port),)).start()

    device = 'cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu'
    compute_type = 'float16' if device == 'cuda' else 'int8'

    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    session = requests.Session()
    session.headers.update({'User-Agent': args.user_agent})

    base_playlist = m3u8.load(args.url)
    highest_bitrate_stream = sorted(base_playlist.playlists, key=lambda x: x.stream_info.bandwidth, reverse=True)[0]
    base_playlist.playlists = PlaylistList([highest_bitrate_stream])

    http_base_url = f'http://{args.bind_address}:{args.bind_port}/'

    modified_base_playlist = copy.deepcopy(base_playlist)
    modified_base_playlist.playlists[0].uri = os.path.join(http_base_url, 'chunklist.m3u8')

    if not args.hard_subs:
        subtitle_list = m3u8.Media(uri=os.path.join(http_base_url, 'subs.m3u8'), type='SUBTITLES', group_id='Subtitle',
                                   language='en', name='English',
                                   forced='NO', autoselect='NO')
        modified_base_playlist.add_media(subtitle_list)
        modified_base_playlist.playlists[0].media += [subtitle_list]

    BASE_PLAYLIST_SER = bytes(modified_base_playlist.dumps(), 'ascii')

    with tempfile.TemporaryDirectory() as chunk_dir:
        prev_cwd = os.getcwd()
        os.chdir(chunk_dir)
        try:
            while True:
                chunk_list = m3u8.load(base_playlist.playlists[0].absolute_uri)

                if chunk_list.target_duration:
                    if int(MAX_TARGET_BUFFER_SECS / chunk_list.target_duration) < len(chunk_list.segments):
                        chunk_list.segments = SegmentList(
                            chunk_list.segments[-int(TARGET_BUFFER_SECS / chunk_list.target_duration):])
                        chunk_list.media_sequence = chunk_list.segments[0].media_sequence

                        if chunk_list.program_date_time:
                            chunk_list.program_date_time = chunk_list.segments[0].current_program_date_time

                with ThreadPool(processes=int(multiprocessing.cpu_count() / 2)) as transcribe_pool:
                    transcribe_pool.map(partial(download_and_transcribe_wrapper, session=session, model=model,
                                                base_uri=chunk_list.base_uri, chunk_dir=chunk_dir,
                                                hard_subs=args.hard_subs, translate=not args.transcribe,
                                                beam_size=args.beam_size, vad_filter=args.vad_filter,
                                                language=args.language), chunk_list.segments)

                current_segments = []
                for segment in chunk_list.segments:
                    segment_name = normalise_chunk_uri(segment.uri)
                    segment.uri = os.path.join(http_base_url, segment_name)
                    current_segments.append(segment_name)

                CHUNK_LIST_SER = bytes(chunk_list.dumps(), 'ascii')

                if not args.hard_subs:
                    for segment in chunk_list.segments:
                        subtitle_name = os.path.splitext(segment.uri)[0] + '.vtt'
                        segment.uri = subtitle_name

                    SUB_LIST_SER = bytes(chunk_list.dumps(), 'ascii')

                for translated_uri, translated_chunk_path in dict(translated_chunk_paths).items():
                    if translated_uri not in current_segments:
                        os.unlink(translated_chunk_path)
                        del translated_chunk_paths[translated_uri]

                if not args.hard_subs:
                    for translated_uri, translated_chunk_path in dict(chunk_to_vtt).items():
                        if os.path.splitext(translated_uri)[0] + '.ts' not in current_segments:
                            del chunk_to_vtt[translated_uri]

                time.sleep(10)
        finally:
            os.chdir(prev_cwd)
