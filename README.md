## Data flow

# Analysis thread
```
AudioSource ──────────► Framer ──────────► Detector ──────────► Mapper ──────────► LatestTarget / OnsetQueue
            float32[n]         float32[4096]        ChordEstimate      ColourState 
            mono samples       window × Hann        root/qual/conf     target HSV  
            None at EOF        None at EOF          +chroma[12]        +onset (pc,hue)
                                                    +onsets (pc)
```

# Consumer thread
```
LatestTarget / OnsetQueue ──────────► Interpolator ──────────► Renderer──────────► Terminal
                          ColourState              ANSI truecolor      print
                          smoothed HSV             pixels              
                          +onset (pc,hue)                         
  
```
---

## Stage 1 — Audio capture (audio/mic.py)

- PortAudio (C/C++) invokes _callback and feeds us `float32[n]`
- Data is written to a ring buffer of `float32[44100]`

---

## Stage 2 — capture (audio/framer.py)

- Framer consumes fixed chunks of default size 2048
- 2048 samples/frame ÷ 44100 samples/sec ≈ 46 ms/frame ≈ 21.5 frames/sec
- Chunks are inserted into a sliding window of size 4096
- For every window we apply a Hann filter

---

## Stage 3 - Chroma detection (detect/detector.py)

- For each frame we compute the chroma (CQT) and onsets (impulse)
- We take the dot product of our chroma and our chord templates to calculate the closest matching chord

---

## Stage 4 - Colour mapper (colour/mapper.py)

- For any given chord, we compute the corresponding hue, saturation, value, intensity
- Chord root determines background colour
- Chord type (major, minor) determines saturation and brightness multipliers
- Mapper confidence also affects saturation
- Loudness also affects brightness and motion
- Onsets are mapped individually to the respective hue of the colour scheme

---

## Stage 5 - Interpolator (colour/interpolator.py)

- Since we don't want to instantly snap between colours, we smoothly interpolate between incoming colours
- We use HSV as each value gives us control over one dimension

---

## Stage 6 - Renderer (render/terminal.py)

- We convert all HSV values to RGB
- Prints ascii to terminal

