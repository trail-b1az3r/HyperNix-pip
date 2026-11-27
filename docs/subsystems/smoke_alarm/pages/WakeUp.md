# wake-up — train a wake word, then listen for it

What [openWakeWord](https://github.com/dscripka/openWakeWord) does,
without using it: take a phrase you choose, gather examples of it and of
everything that is not it, train a small classifier on log-mel frames,
and run that classifier over a sliding window of live audio.

```bash
hypernix wakeup train "hey nix" --positives ./takes --negatives ./ambient
hypernix wakeup listen --model hey-nix.pt
```

## Three ways to get examples, and they mix

| Source | Function | Cost |
|---|---|---|
| **Your own voice, live** | `record_examples` | Best quality per sample, slowest to gather |
| **A folder you already have** | `examples_from_folder` | WAV, MP3, FLAC, fragmented MP3 |
| **1–4 TTS voices, overnight** | `synthesize_examples` | No human input, least like a real microphone |

They are *meant* to be combined. A model trained only on TTS learns what
synthesised speech sounds like, which is not the task — a handful of real
recordings alongside a few hundred synthetic ones is worth far more than
either alone, and `build_dataset` takes all three at once.

```bash
# Overnight: no input from you.
hypernix wakeup train "hey nix" --tts --voices 3 --negatives ./ambient

# Twenty of your own takes, then TTS on top.
hypernix wakeup record "hey nix" -o ./takes -n 20
hypernix wakeup train "hey nix" --positives ./takes --tts --negatives ./ambient
```

More than four voices is refused rather than silently truncated: past
that the marginal voice costs hours and adds less than a handful of real
recordings would.

## Fragmented MP3

A stream cut into pieces is a set of files that decode individually but
only mean something in order. `--fragments` joins them, sorted
**naturally** — `part2` before `part10`, not after it.

That is not pedantry. Lexicographic ordering puts `part10` second, so a
reassembled stream has its middle in the wrong place, and the result is
audio that sounds *almost* right — the worst kind of wrong for training
data, because nothing about it looks like a bug.

## The processing, and why it is not optional

Every example is augmented: gain, time shift, speed, background noise
and a cheap reverb.

This is not padding the dataset. A wake word is heard from across a room,
over a fan, at whatever distance and volume the speaker happens to be —
and a model trained on clean centred clips learns *"clean and centred"*
as part of the phrase, then does not fire in a kitchen. The augmentation
is where most of the real-world accuracy comes from.

Negatives are augmented too. Clean negatives against noisy positives
teaches the model to detect noise.

| Knob | Default | Why |
|---|---|---|
| `gain_db` | −12 to +6 | Distance and mic gain. Clipped at full scale — a gain past 1.0 wraps when written to 16-bit, which sounds like a click and teaches the model the phrase contains one. |
| `shift` | 0.35 | The phrase does not always start at frame zero. |
| `speed` | 0.9–1.1 | People say it faster when they are used to it. |
| `snr_db` | 5–30 | Mixed from your own ambient recordings. |
| `reverb_probability` | 0.3 | Three decaying taps — a hint of a room, not a room model. Convolving with a measured impulse response would be better and would mean shipping impulse responses. |

## Negatives matter more than positives

A wake-word model is mostly a **rejector**: it will hear a thousand hours
of not-the-phrase for every utterance of it. The dataset says so before
you train on it:

```
  ! Only 1.0 negatives per positive. A model trained near 1:1 fires at
    the television — aim for 5-20x, from ambient recordings and other
    speech.
```

`--strict` turns those warnings into an error. Off by default, because
someone experimenting with twenty clips should be allowed to; on for
anything that will be deployed, because "it trained fine" is not the same
as "it will work in a kitchen".

## The model

Conv over the mel axis (a phoneme is a local pattern in frequency), then
a GRU over time (the phrase is an ordering of them), then one output.
Small on purpose: this runs continuously on whatever you have, and a
model that needs a GPU to decide whether someone said two words is not a
wake-word model, it is a reason not to use one.

Deliberately **not** bidirectional. It would score better offline and is
useless here — the detector has to answer before the utterance ends.

The loss is weighted by the class imbalance. Unweighted, a model on a 5:1
dataset learns that answering "no" is right 83% of the time, and the
accuracy number looks fine while the thing never fires. `train_wakeword`
reports precision and recall for the same reason.

## Detection

```python
detector = load_detector("hey-nix.pt")
while True:
    hit = detector.push(microphone.read(1600))
    if hit:
        print(hit["confidence"])
```

`push` **returns** rather than calls back — this module has no business
ringing a bell. It keeps a ring of the last `window_seconds` and scores
as often as `hop_seconds` allows, so cost is bounded by the hop and not
by how small the caller's chunks happen to be.

After a detection it holds off for `refractory_seconds`. Without that,
one utterance fires on every overlapping window it appears in and the
caller gets six wakes for one word.

The default threshold is 0.75, deliberately high: a false accept is a
device waking in a silent room, which people notice far more than a
missed word they can simply repeat.

## The config travels with the weights

A wake-word model is useless without the exact mel settings it was
trained on — a different hop or band count and the model sees a different
spectrum. `save_wakeword` writes them into the checkpoint, because
keeping them in a separate file is how that pairing gets broken.

## Reading audio

`hypernix.audio.audiofile` decodes WAV in pure Python — it is a format,
not a codec, and shipping a dependency for it would be silly. Everything
else goes to ffmpeg if installed, or `soundfile`/`librosa` if they are.
With none of those, a file is skipped with a message naming what would
let it be read, rather than an ImportError halfway through a folder.

One corrupt file does not lose the folder: it is skipped and reported.
Finding out which one it was after an hour of gathering is worse than
being told at the time.

## See also

- [Workshop](Workshop.md) — the TTS and ASR engines this borrows for synthesis
- [CLI](CLI.md#wakeup) — every flag
