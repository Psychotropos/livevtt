# LiveVTT

LiveVTT is a tool for live transcription of streaming audio/video content, providing real-time subtitles in WebVTT format.

## Description

LiveVTT allows you to transcribe live audio/video streams and generate WebVTT subtitles. It supports various features such as model selection, CUDA utilization, silence filtering, and more.

## Usage

```bash
livevtt -u <URL> [-s] [-l <BIND_ADDRESS>] [-p <BIND_PORT>] [-m <MODEL>] [-b <BEAM_SIZE>] [-c <USE_CUDA>] [-t <TRANSLATE>] [-vf <VAD_FILTER>] [-la <LANGUAGE>] [-ua <USER_AGENT>]
```

### Arguments

- `-u, --url`: **[Required]** URL of the live audio/video stream.
- `-s, --hard-subs`: Set if you want the subtitles to be baked into the stream itself.
- `-l, --bind-address`: The IP address to bind to (defaults to 127.0.0.1).
- `-p, --bind-port`: The port to bind to (defaults to 8000).
- `-m, --model`: Whisper model to use (defaults to large).
- `-b, --beam-size`: Beam size to use (defaults to 5).
- `-c, --use-cuda`: Use CUDA where available. Defaults to true.
- `-t, --translate`: If set to false, only provides a transcription for the given stream. Defaults to true.
- `-vf, --vad-filter`: Whether to utilize the Silero VAD model to try and filter out silences. Defaults to false.
- `-la, --language`: The original language of the stream, if known/not multi-lingual. Can be left unset.
- `-ua, --user-agent`: User agent to use to retrieve playlists/stream chunks (defaults to 'VLC/3.0.18 LibVLC/3.0.18').

## Accessing Transcribed Stream

Once the program is running, you can access the transcribed and/or translated stream at the following URL:

```
http://127.0.0.1:8000/playlist.m3u8
```

This URL may vary based on the bind address and port provided via the command-line options.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Psychotropos/livevtt.git
   ```

2. Navigate to the directory:

   ```bash
   cd livevtt
   ```

3. Install dependencies:

   - For general installation:

   ```bash
   pip install -r requirements.txt
   ```

   - For CUDA support on Windows:

   ```bash
   pip install -r requirements-cuda-win.txt
   ```

## Examples

1. Transcribe a live audio/video stream with default settings:

   ```bash
   livevtt -u <URL>
   ```

2. Transcribe a live audio/video stream and embed subtitles:

   ```bash
   livevtt -u <URL> -s
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
