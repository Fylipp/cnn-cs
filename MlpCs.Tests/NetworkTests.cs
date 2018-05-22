using System;
using System.Linq;
using Fylipp.MlpCs;
using NUnit.Framework;

namespace MlpCs.Tests {
    [TestFixture]
    public class NetworkTests {
        [Test]
        public void TestFactory1() {
            var dimensions = new[] {4, 3, 3, 2};

            var network = Network.Generate(dimensions);

            Assert.AreEqual(dimensions.Length, network.LayerCount);
            Assert.AreEqual(dimensions.Length - 1, network.NonInputLayerCount);
            Assert.AreEqual(dimensions[0], network.InputDimension);
            AssertNetworkWeightsInBounds(-1, 1, network);

            for (var i = 0; i < network.NonInputLayerCount; i++) {
                Assert.AreEqual(dimensions[i + 1], network.NonInputLayers[i].Dimension);
                Assert.AreEqual(dimensions[i], network.NonInputLayers[i].InputDimension);
            }
        }

        [Test]
        public void TestFactory3() {
            var dimensions = new[] {4, 3, 3, 2};
            const double min = -4;
            const double max = -2.5;

            var network = Network.Generate(dimensions, min, max);

            Assert.AreEqual(dimensions.Length, network.LayerCount);
            Assert.AreEqual(dimensions.Length - 1, network.NonInputLayerCount);
            Assert.AreEqual(dimensions[0], network.InputDimension);
            AssertNetworkWeightsInBounds(min, max, network);

            for (var i = 0; i < network.NonInputLayerCount; i++) {
                Assert.AreEqual(dimensions[i + 1], network.NonInputLayers[i].Dimension);
                Assert.AreEqual(dimensions[i], network.NonInputLayers[i].InputDimension);
            }
        }

        [Test]
        public void TestFactory4() {
            var dimensions = new[] {4, 3, 3, 2};
            const double min = -4;
            const double max = -2.5;

            var network = Network.Generate(dimensions, min, max, Neuron.IdentityActivation);

            Assert.AreEqual(dimensions.Length, network.LayerCount);
            Assert.AreEqual(dimensions.Length - 1, network.NonInputLayerCount);
            Assert.AreEqual(dimensions[0], network.InputDimension);
            AssertNetworkWeightsInBounds(min, max, network);

            for (var i = 0; i < network.NonInputLayerCount; i++) {
                Assert.AreEqual(dimensions[i + 1], network.NonInputLayers[i].Dimension);
                Assert.AreEqual(dimensions[i], network.NonInputLayers[i].InputDimension);
            }

            AssertForNeurons(network,
                neuron => TestUtils.AssertActivationFunctionsEqual(Neuron.IdentityActivation, neuron.Activation));
        }

        [Test]
        public void TestCalculateWithIdentityActivationFunction() {
            var layer1 = new Layer(
                new Neuron(Neuron.IdentityActivation, -3.1),
                new Neuron(Neuron.IdentityActivation, 4.6));

            var layer2 = new Layer(
                new Neuron(Neuron.IdentityActivation, 2.0, 3.0)
            );

            var network = new Network(1, layer1, layer2);

            var output = network.Calculate(10);
            var expectedOutput = new[] {10 * -3.1 * 2 + 10 * 4.6 * 3};

            Assert.AreEqual(expectedOutput, output);
        }

        [Test]
        public void TestCalculateWithDefaultActivationFunction() {
            var layer1 = new Layer(
                new Neuron(-3.1),
                new Neuron(4.6));

            var layer2 = new Layer(
                new Neuron(2.0, 3.0)
            );

            var network = new Network(1, layer1, layer2);

            var output = network.Calculate(10);

            var l1_n1 = Neuron.DefaultActivation(10 * -3.1);
            var l1_n2 = Neuron.DefaultActivation(10 * 4.6);
            var l2_n1 = Neuron.DefaultActivation(l1_n1 * 2 + l1_n2 * 3);
            var expectedOutput = new[] {l2_n1};

            Assert.AreEqual(expectedOutput, output);
        }

        [Test]
        public void TestCalculateWithCustomActivationFunction() {
            double Activation(double x) => x + 3;

            var layer1 = new Layer(
                new Neuron(Activation, -3.1),
                new Neuron(Activation, 4.6));

            var layer2 = new Layer(
                new Neuron(Activation, 2.0, 3.0)
            );

            var network = new Network(1, layer1, layer2);

            var output = network.Calculate(10);
            var l1_n1 = Activation(10 * -3.1);
            var l1_n2 = Activation(10 * 4.6);
            var l2_n1 = Activation(l1_n1 * 2 + l1_n2 * 3);
            var expectedOutput = new[] {l2_n1};

            Assert.AreEqual(expectedOutput, output);
        }

        public static void AssertNetworkWeightsInBounds(double min, double max, Network network) {
            foreach (var layer in network.NonInputLayers) {
                LayerTests.AssertLayerWeightsInBounds(min, max, layer);
            }
        }

        public static void AssertForNeurons(Network network, Predicate<Neuron> predicate) =>
            AssertForNeurons(network, neuron => Assert.IsTrue(predicate(neuron)));

        public static void AssertForNeurons(Network network, Action<Neuron> assertion) {
            foreach (var layer in network.NonInputLayers) {
                LayerTests.AssertForNeurons(layer, assertion);
            }
        }

        public static void AssertForLayers(Network network, Predicate<Layer> predicate) =>
            AssertForLayers(network, layer => Assert.IsTrue(predicate(layer)));

        public static void AssertForLayers(Network network, Action<Layer> assertion) {
            foreach (var layer in network.NonInputLayers) {
                assertion(layer);
            }
        }
    }
}
