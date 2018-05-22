using System;
using System.Linq;
using JetBrains.Annotations;

namespace Fylipp.MlpCs {
    /// <summary>
    /// A network of neurons composed into layers.
    /// </summary>
    public class Network {
        /// <summary>
        /// The non-input layers of the network.
        /// </summary>
        [NotNull]
        public Layer[] NonInputLayers { get; }

        /// <summary>
        /// The amount of layers.
        /// </summary>
        public int NonInputLayerCount => NonInputLayers.Length;

        /// <summary>
        /// The amount of layers, including the input layer.
        /// </summary>
        public int LayerCount => NonInputLayerCount + 1;

        /// <summary>
        /// The dimension of the input.
        /// </summary>
        public int InputDimension { get; }

        /// <summary>
        /// Creates a network with the given non-input layers.
        /// </summary>
        /// <param name="inputDimension">The dimension of the input</param>
        /// <param name="nonInputLayers">The non-input layers</param>
        public Network(int inputDimension, [NotNull] params Layer[] nonInputLayers) {
            InputDimension = inputDimension;
            NonInputLayers = nonInputLayers;

            if (InputDimension < 1) {
                throw new ArgumentException("Network input dimension must be at least 1");
            }

            if (NonInputLayers.Any(layer => layer == null)) {
                throw new NullReferenceException("Layer may not be null");
            }
        }

        /// <summary>
        /// Generates a network with random weights.
        /// The resulting network will not explicitly contain the first layer since this layer represents the
        /// input values and these are passed to the network explicitly.
        /// </summary>
        /// <param name="layerDimensions">The amount of neurons in each layer</param>
        /// <param name="minimumWeight">The lowest value for a weight</param>
        /// <param name="maximumWeight">The highest value for a weight</param>
        /// <param name="activation">The activation function of the neurons</param>
        [NotNull]
        public static Network Generate([NotNull] int[] layerDimensions, double minimumWeight = -1.0,
            double maximumWeight = 1.0,
            [CanBeNull] Neuron.ActivationFunction activation = null) {
            if (layerDimensions.Length == 0) {
                throw new ArgumentException("There must be at least one layer", nameof(layerDimensions));
            }

            var layers = new Layer[layerDimensions.Length - 1];

            for (var layerIndex = 1; layerIndex < layerDimensions.Length; layerIndex++) {
                var dimension = layerDimensions[layerIndex];
                var inputDimension = layerDimensions[layerIndex - 1];

                var layer = Layer.Generate(dimension, inputDimension, minimumWeight, maximumWeight, activation);

                layers[layerIndex - 1] = layer;
            }

            return new Network(layerDimensions[0], layers);
        }

        /// <summary>
        /// Calculates the output of the network.
        /// </summary>
        /// <param name="inputValues">The input values</param>
        /// <returns>The aggregated output of the network.</returns>
        [NotNull]
        public double[] Calculate([NotNull] params double[] inputValues) =>
            NonInputLayers.Aggregate(inputValues, (values, layer) => layer.Calculate(values));
    }
}
