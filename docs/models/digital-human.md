# Digital Human & Maxine Models

> Facial animation, gaze correction, and audio processing for digital humans.

## Canonical URL

`https://docs.nvidia.com/nim/index.html` (Digital Human and Maxine sections)

## Models

### Audio2Face-3D
| Feature | Description |
|---|---|
| **Input** | Audio stream (speech) |
| **Output** | 3D facial animation blend shapes |
| **Use Case** | Animate 3D avatars from speech in real-time |
| **Integration** | Game engines (Unreal, Unity), 3D tools |

### Audio2Face-2D
| Feature | Description |
|---|---|
| **Input** | Audio stream + reference face image |
| **Output** | 2D facial animation video |
| **Use Case** | Lip-sync video for 2D characters or talking heads |

### Eye Contact
| Feature | Description |
|---|---|
| **Input** | Video stream |
| **Output** | Gaze-corrected video stream |
| **Use Case** | Fix eye contact in video calls (look at camera) |
| **Real-time** | Yes, designed for live video |

### Studio Voice
| Feature | Description |
|---|---|
| **Input** | Noisy audio |
| **Output** | Clean, studio-quality audio |
| **Use Case** | Remove background noise, enhance voice clarity |
| **Real-time** | Yes, designed for live audio |

## Digital Human Pipeline

```
Text → TTS NIM → Audio2Face-3D → 3D Engine → Video Output
                                      ↑
                              Avatar Model
```

1. Generate speech from text using TTS NIM
2. Generate facial animation from speech using Audio2Face
3. Apply animation to 3D avatar in game engine
4. Render final video output

## Use Cases

| Application | Models Used |
|---|---|
| Virtual assistants | TTS + Audio2Face-3D |
| Gaming NPCs | TTS + Audio2Face-3D |
| Video conferencing | Eye Contact + Studio Voice |
| Content creation | Audio2Face-2D + TTS |
| Customer service avatars | TTS + Audio2Face-3D |
| Telepresence | Eye Contact + Studio Voice |

## Deployment

All digital human models are available as NIM containers, deployable on NVIDIA GPUs for real-time inference.
