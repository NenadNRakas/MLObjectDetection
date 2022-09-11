using Microsoft.ML.Data;

namespace MLObjectDetection.DataStructures
{
    public class ImageNetDetection
    {
        [ColumnName("grid")]
        public float[] PredictedLabels;
    }
}
