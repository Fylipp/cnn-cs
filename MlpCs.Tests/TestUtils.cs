using Fylipp.MlpCs;
using NUnit.Framework;

namespace MlpCs.Tests {
    public static class TestUtils {
        public static void AssertInBounds(double min, double max, double value) =>
            Assert.That(() => value >= min && value <= max);

        public static void AssertActivationFunctionsEqual(Neuron.ActivationFunction left,
            Neuron.ActivationFunction right, params double[] values) {
            foreach (var value in values) {
                Assert.AreEqual(left(value), right(value));
            }
        }
    }
}
