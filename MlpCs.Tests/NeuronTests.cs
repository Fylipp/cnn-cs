using System;
using Fylipp.MlpCs;
using NUnit.Framework;

namespace MlpCs.Tests {
    [TestFixture]
    public class NeuronTests {
        [Test]
        public void TestFactory1() {
            const int degree = 15;

            var neuron = Neuron.Generate(degree);

            Assert.AreEqual(degree, neuron.Degree);
            TestUtils.AssertActivationFunctionsEqual(Neuron.DefaultActivation, neuron.Activation);
            AssertNeuronWeightsInBounds(-1, 1, neuron);
        }

        [Test]
        public void TestFactory3() {
            const int degree = 15;
            const double min = -2.7;
            const int max = 9;

            var neuron = Neuron.Generate(degree, min, max);

            Assert.AreEqual(degree, neuron.Degree);
            TestUtils.AssertActivationFunctionsEqual(Neuron.DefaultActivation, neuron.Activation);
            AssertNeuronWeightsInBounds(min, max, neuron);
        }

        [Test]
        public void TestFactory4() {
            const int degree = 19;
            const int min = 3;
            const int max = 14;

            var neuron = Neuron.Generate(degree, min, max, Neuron.IdentityActivation);

            Assert.AreEqual(degree, neuron.Degree);
            TestUtils.AssertActivationFunctionsEqual(Neuron.IdentityActivation, neuron.Activation);
            AssertNeuronWeightsInBounds(min, max, neuron);
        }

        [Test]
        public void TestConstructor1() {
            var weights = new[] {0.12, Math.PI, 3.456 * Math.Pow(10, 4)};

            var neuron = new Neuron(weights);

            Assert.AreEqual(weights, neuron.Weights);
            Assert.IsNotNull(neuron.Activation);
            Assert.AreEqual(weights.Length, neuron.Degree);
        }

        [Test]
        public void TestConstructor2() {
            var neuron = new Neuron(Neuron.IdentityActivation, 0.018, Math.E, 7.1945 * Math.Pow(10, 9));

            Assert.AreEqual(new[] {0.018, Math.E, 7.1945 * Math.Pow(10, 9)}, neuron.Weights);
            Assert.IsNotNull(neuron.Activation);
            TestUtils.AssertActivationFunctionsEqual(Neuron.IdentityActivation, neuron.Activation);
        }

        [Test]
        public void TestCalculateWithIdentityActivationFunction() {
            var neuron = new Neuron(Neuron.IdentityActivation, 2, .5, -1);

            var output = neuron.Calculate(4, -7, 19.5);
            var expectedOutput = 2 * 4 + .5 * -7 + -1 * 19.5;

            Assert.AreEqual(expectedOutput, output);
        }

        [Test]
        public void TestCalculateWithDefaultActivationFunction() {
            var neuron = new Neuron(2, .5, -1);

            var output = neuron.Calculate(4, -7, 19.5);
            var expectedOutput = Neuron.DefaultActivation(2 * 4 + .5 * -7 + -1 * 19.5);

            Assert.AreEqual(expectedOutput, output);
        }

        [Test]
        public void TestCalculateWithCustomActivationFunction() {
            double Activation(double x) => x / 2;

            var neuron = new Neuron(Activation, 2, .5, -1);

            var output = neuron.Calculate(4, -7, 19.5);
            var expectedOutput = Activation(2 * 4 + .5 * -7 + -1 * 19.5);

            Assert.AreEqual(expectedOutput, output);
        }

        public static void AssertNeuronWeightsInBounds(double min, double max, Neuron neuron) {
            foreach (var weight in neuron.Weights) {
                TestUtils.AssertInBounds(min, max, weight);
            }
        }
    }
}
