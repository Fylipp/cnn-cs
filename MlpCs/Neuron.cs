using System;
using System.Linq;
using JetBrains.Annotations;

namespace Fylipp.MlpCs {
    /// <summary>
    /// A neuron that has a set of weights and an activation function.
    /// </summary>
    public class Neuron {
        private static readonly Random Random = new Random();

        /// <summary>
        /// An activation function maps the weighted sum of values to the neuron output.
        /// Identitfy function (x -> x) are disadvantageous for learning systems.
        /// </summary>
        /// <param name="weightedSum">The weighted sum of the neuron</param>
        public delegate double ActivationFunction(double weightedSum);

        /// <summary>
        /// The default activation function is a sigmoid logistical function.
        /// </summary>
        [Pure]
        public static double DefaultActivation(double weightedSum) => 1 / (1 + Math.Exp(-weightedSum));

        /// <summary>
        /// An activation function that does not transform the input.
        /// </summary>
        [Pure]
        public static double IdentityActivation(double weightedSum) => weightedSum;

        /// <summary>
        /// The weights of the neuron.
        /// </summary>
        public double[] Weights { get; }

        /// <summary>
        /// The amount of neuron inputs.
        /// </summary>
        public int Degree => Weights.Length;

        /// <summary>
        /// The activation function of the neuron.
        /// </summary>
        public ActivationFunction Activation { get; }

        /// <summary>
        /// Creates a neuron with weights and an activation function.
        /// </summary>
        public Neuron([CanBeNull] ActivationFunction activation, [NotNull] params double[] weights) {
            Weights = weights;
            Activation = activation ?? DefaultActivation;
        }

        /// <summary>
        /// Creates a neuron with weights and the default activation function.
        /// </summary>
        public Neuron([NotNull] params double[] weights) : this(null, weights) { }

        /// <summary>
        /// Generates a neuron with random weights.
        /// </summary>
        /// <param name="degree">The amount of input weights</param>
        /// <param name="minimumWeight">The lowest value for a weight</param>
        /// <param name="maximumWeight">The highest value for a weight</param>
        /// <param name="activation">The activation function of the neurons</param>
        public static Neuron Generate(int degree,
            double minimumWeight = -1, double maximumWeight = 1, [CanBeNull] ActivationFunction activation = null) {
            if (degree < 1) {
                throw new ArgumentException("The degree must be at least 1", nameof(degree));
            }

            if (minimumWeight > maximumWeight) {
                throw new ArgumentException("The minimum weight must not exceed the maximum weight",
                    nameof(minimumWeight));
            }

            var weights = new double[degree];

            for (var i = 0; i < degree; i++) {
                // Produce a random double-precision floating-point value in the specified interval
                weights[i] = Random.NextDouble() * (maximumWeight - minimumWeight) + minimumWeight;
            }

            return new Neuron(activation, weights);
        }

        /// <summary>
        /// Calculates the output value of the neuron.
        /// </summary>
        /// <param name="inputValues">The input layer</param>
        /// <returns>The activated sum of weighted inputs</returns>
        public double Calculate([NotNull] params double[] inputValues) => Activation(WeightedSum(inputValues));

        /// <summary>
        /// Calculates a weighted sum of the input.
        /// </summary>
        /// <param name="inputValues">The input layer</param>
        private double WeightedSum([NotNull] double[] inputValues) {
            if (inputValues.Length != Degree) {
                throw new ArgumentException("The amount of input values must match the neurons degree");
            }

            var weighted = new double[Degree];

            for (var i = 0; i < Degree; i++) {
                weighted[i] = inputValues[i] * Weights[i];
            }

            return weighted.Aggregate((l, r) => l + r);
        }
    }
}
