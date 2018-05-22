using System;
using System.Linq;
using JetBrains.Annotations;

namespace Fylipp.MlpCs {
    /// <summary>
    /// A layer of neurons.
    /// </summary>
    public class Layer {
        /// <summary>
        /// The neurons of the layer.
        /// </summary>
        [NotNull]
        public Neuron[] Neurons { get; }

        /// <summary>
        /// The amount of neurons in the layer.
        /// </summary>
        public int Dimension => Neurons.Length;

        /// <summary>
        /// The dimension of the layer input.
        /// </summary>
        public int InputDimension { get; }

        /// <summary>
        /// Creates a layer with the given neurons.
        /// </summary>
        public Layer([NotNull] params Neuron[] neurons) {
            Neurons = neurons;

            if (Neurons.Length < 1) {
                throw new ArgumentException("Layer must contain at least one neuron");
            }

            if (Neurons.Any(neuron => neuron == null)) {
                throw new NullReferenceException("Neuron may not be null");
            }

            var degree = Neurons[0].Degree;
            if (Neurons.Any(neuron => neuron.Degree != degree)) {
                throw new ArgumentException("Inconsistent neuron degrees");
            }

            InputDimension = degree;
        }

        /// <summary>
        /// Generates a layer with random weights.
        /// </summary>
        /// <param name="dimension">The amount of neurons in the layer</param>
        /// <param name="inputDimension">The amount of neurons in the previous layer</param>
        /// <param name="minimumWeight">The lowest value for a weight</param>
        /// <param name="maximumWeight">The highest value for a weight</param>
        /// <param name="activation">The activation function of the neurons</param>
        [NotNull]
        public static Layer Generate(int dimension, int inputDimension, double minimumWeight = -1.0,
            double maximumWeight = 1, [CanBeNull] Neuron.ActivationFunction activation = null) {
            if (dimension < 1) {
                throw new ArgumentException("The layer dimension must be at least 1", nameof(dimension));
            }

            var neurons = new Neuron[dimension];

            for (var i = 0; i < dimension; i++) {
                neurons[i] = Neuron.Generate(inputDimension, minimumWeight, maximumWeight, activation);
            }

            return new Layer(neurons);
        }

        /// <summary>
        /// Calculates the layer output values.
        /// </summary>
        /// <param name="inputValues">The input layer values</param>
        /// <returns>The mapped output of the layer</returns>
        [NotNull]
        public double[] Calculate([NotNull] params double[] inputValues) =>
            Neurons.Select(n => n.Calculate(inputValues)).ToArray();
    }
}
