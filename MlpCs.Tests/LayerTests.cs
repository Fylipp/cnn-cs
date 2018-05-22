using System;
using System.Linq;
using Fylipp.MlpCs;
using NUnit.Framework;

namespace MlpCs.Tests {
    [TestFixture]
    public class LayerTests {
        [Test]
        public void TestFactory2() {
            const int dimension = 5;
            const int inputDimension = 7;

            var layer = Layer.Generate(dimension, inputDimension);

            Assert.AreEqual(dimension, layer.Dimension);
            Assert.AreEqual(dimension, layer.Neurons.Length);
            Assert.AreEqual(inputDimension, layer.InputDimension);
            AssertLayerWeightsInBounds(-1, 1, layer);
            AssertForNeurons(layer, neuron => neuron.Degree == inputDimension);
            AssertForNeurons(layer,
                neuron => TestUtils.AssertActivationFunctionsEqual(Neuron.DefaultActivation, neuron.Activation));
        }

        [Test]
        public void TestFactory4() {
            const int dimension = 3;
            const int inputDimension = 12;
            const int min = -10;
            const int max = 27;

            var layer = Layer.Generate(dimension, inputDimension, min, max);

            Assert.AreEqual(dimension, layer.Dimension);
            Assert.AreEqual(dimension, layer.Neurons.Length);
            Assert.AreEqual(inputDimension, layer.InputDimension);
            AssertLayerWeightsInBounds(min, max, layer);
            AssertForNeurons(layer, neuron => neuron.Degree == inputDimension);
            AssertForNeurons(layer,
                neuron => TestUtils.AssertActivationFunctionsEqual(Neuron.DefaultActivation, neuron.Activation));
        }

        [Test]
        public void TestFactory5() {
            const int dimension = 8;
            const int inputDimension = 4;
            const int min = 6;
            const int max = 7;

            var layer = Layer.Generate(dimension, inputDimension, min, max, Neuron.IdentityActivation);

            Assert.AreEqual(dimension, layer.Dimension);
            Assert.AreEqual(dimension, layer.Neurons.Length);
            Assert.AreEqual(inputDimension, layer.InputDimension);
            AssertLayerWeightsInBounds(min, max, layer);
            AssertForNeurons(layer, neuron => neuron.Degree == inputDimension);
            AssertForNeurons(layer,
                neuron => TestUtils.AssertActivationFunctionsEqual(Neuron.IdentityActivation, neuron.Activation));
        }

        [Test]
        public void TestConstructor() {
            var neurons = new[] {Neuron.Generate(2), Neuron.Generate(2), Neuron.Generate(2)};

            var layer = new Layer(neurons);

            Assert.AreEqual(neurons, layer.Neurons);
            Assert.AreEqual(neurons.Length, layer.Dimension);
        }

        [Test]
        public void TestCalculateWithIdentityActivationFunction() {
            var neurons = new[] {
                new Neuron(Neuron.IdentityActivation, new[] {1.5, -1}),
                new Neuron(Neuron.IdentityActivation, new[] {3, 6.5})
            };

            var layer = new Layer(neurons);

            var output = layer.Calculate(new[] {4, -3.2});
            var expectedOutput = new[] {4 * 1.5 + -3.2 * -1, 4 * 3 + -3.2 * 6.5};

            Assert.AreEqual(expectedOutput, output);
        }

        [Test]
        public void TestCalculateWithDefaultActivationFunction() {
            var neurons = new[] {new Neuron(null, new[] {1.5, -1}), new Neuron(null, new[] {3, 6.5})};

            var layer = new Layer(neurons);

            var output = layer.Calculate(4, -3.2);
            var expectedOutput = new[] {4 * 1.5 + -3.2 * -1, 4 * 3 + -3.2 * 6.5}.Select(Neuron.DefaultActivation);

            Assert.AreEqual(expectedOutput, output);
        }

        [Test]
        public void TestCalculateWithCustomActivationFunction() {
            double Activation(double x) => 2 * x;
            var neurons = new[] {new Neuron(Activation, 1.5, -1), new Neuron(Activation, 3, 6.5)};

            var layer = new Layer(neurons);

            var output = layer.Calculate(4, -3.2);
            var expectedOutput = new[] {4 * 1.5 + -3.2 * -1, 4 * 3 + -3.2 * 6.5}.Select(Activation);

            Assert.AreEqual(expectedOutput, output);
        }

        public static void AssertLayerWeightsInBounds(double min, double max, Layer layer) {
            foreach (var neuron in layer.Neurons) {
                NeuronTests.AssertNeuronWeightsInBounds(min, max, neuron);
            }
        }

        public static void AssertForNeurons(Layer layer, Predicate<Neuron> predicate) =>
            AssertForNeurons(layer, neuron => Assert.IsTrue(predicate(neuron)));

        public static void AssertForNeurons(Layer layer, Action<Neuron> assertion) {
            foreach (var neuron in layer.Neurons) {
                assertion(neuron);
            }
        }
    }
}
