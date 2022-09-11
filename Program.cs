using System.Drawing;
using System.Drawing.Drawing2D;
using Microsoft.ML;
using MLObjectDetection;
using MLObjectDetection.DataStructures;
using MLObjectDetection.YoloParser;

//namespace MLObjectDetection { //internal class Program { static void Main(string[] args) { Console.WriteLine("Hello, World!"); }}}

var assetsRelativePath = @"../../../assets";
string assetsPath = GetAbsolutePath(assetsRelativePath);
var modelFilePath = Path.Combine(assetsPath, "Model", "TinyYolo2_model.onnx");
var imagesFolder = Path.Combine(assetsPath, "images");
var outputFolder = Path.Combine(assetsPath, "images", "output");
MLContext mlContext = new MLContext();
//var model = mlContext.Model;
try
{
    // Load data
    IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
    IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);
    // Create instance
    var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);
    // Score data
    IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);
    // Post-process
    YoloOutputParser parser = new YoloOutputParser();

    var boundingBoxes =
        probabilities
        .Select(probability => parser.ParseOutputs(probability))
        .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

    // Draw bounding boxes for detected objects in each of the images
    for (var i = 0; i < images.Count(); i++)
    {
        string imageFileName = images.ElementAt(i).Label;
        IList<YoloBoundingBox> detectedObjects = boundingBoxes.ElementAt(i);

        DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);

        LogDetectedObjects(imageFileName, detectedObjects);
    }
}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}
Console.WriteLine("========= End of Process..Hit any Key ========");
string GetAbsolutePath(string relativePath)
{

    FileInfo _dataRoot = new(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;

    string fullPath = Path.Combine(assemblyFolderPath, relativePath);

    return fullPath;
}
void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
{
    // Load
    Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));
    // Get dimestions
    var originalImageHeight = image.Height;
    var originalImageWidth = image.Width;
    foreach (var box in filteredBoundingBoxes)
    {

        // Get dimensions of bounding boxes
        var x = (uint)Math.Max(box.Dimensions.X, 63);
        var y = (uint)Math.Max(box.Dimensions.Y, 63);
        var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
        var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);
        // Return to original size
        x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
        y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
        width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
        height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;
        // Text templete
        string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";
        // Convert to graphics object
        using (Graphics thumbnailGraphic = Graphics.FromImage(image))
        {
            // Tune the graphics settings
            thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
            thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
            thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;
            // Define Text font and color
            Font drawFont = new Font("Arial", 12, FontStyle.Bold);
            SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
            SolidBrush fontBrush = new SolidBrush(Color.Black);
            Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);
            // Define BoundingBox color
            Pen pen = new Pen(box.BoxColor, 3.2f);
            SolidBrush colorBrush = new SolidBrush(box.BoxColor);
            // Accessibility options
            thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
            // Draw text
            thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);
            // Draw bounding box
            thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
        }
    }
    // Save images
    if (!Directory.Exists(outputImageLocation))
    {
        Directory.CreateDirectory(outputImageLocation);
    }
    image.Save(Path.Combine(outputImageLocation, imageName));
}
void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
{

    Console.WriteLine($".....The objects in the image {imageName} are detected as below....");
    foreach (var box in boundingBoxes)
    {
        Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
    }
    Console.WriteLine("");
}