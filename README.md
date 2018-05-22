# mlp-cs

[![Build Status](https://travis-ci.org/Fylipp/mlp-cs.svg?branch=master)](https://travis-ci.org/Fylipp/mlp-cs)

A simple multilayer perceptron library for C#.

Originally made to be used within [Unity3D](https://unity3d.com), but useable anywhere.
Extracted from one of my school projects ([flappy-pos-logic](https://github.com/Fylipp/flappy-pos-logic)).

## Usage

```csharp
using Fylipp.MlpCs;

var network = Network.Generate(new []{2, 2, 1});
var output = network.Calculate(3.5, -1.2); // output = double[]
```

## Integration

Download the latest release from the [releases tab](https://github.com/Fylipp/mlp-cs/releases).

To use it in Unity3D drag the `.dll` into the `Assets` folder.

## License

MIT.
