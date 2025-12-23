using System;
using GemBox.Spreadsheet;
using GemBox.Spreadsheet.Charts;

class Program
{
    static void Main()
    {
        // If using the Professional version, put your serial key below.
        SpreadsheetInfo.SetLicense("FREE-LIMITED-KEY");

        // Use a single DateTime.Now so filename is consistent.
        var now = DateTime.Now;
        string fileName = $"Earth–{now:HH}h{now:mm}m.xlsx";

        var workbook = new ExcelFile();
        var worksheet = workbook.Worksheets.Add("Breakdown");

        // Header
        worksheet.Cells[0, 0].Value = "Continents";
        worksheet.Cells[0, 1].Value = "Area (km2)";

        // Make header bold
        var headerRange = worksheet.Cells.GetSubrangeAbsolute(0, 0, 0, 1);
        headerRange.Style.Font.Weight = ExcelFont.BoldWeight;

        // Approximate continent areas (example values; adjust if needed)
        string[] continents = new[]
        {
            "Asia",
            "Africa",
            "North America",
            "South America",
            "Antarctica",
            "Europe",
            "Oceania"
        };

        double[] areas = new[]
        {
            44579000d, // Asia
            30370000d, // Africa
            24709000d, // North America
            17840000d, // South America
            14200000d, // Antarctica
            10180000d, // Europe
            8525989d   // Oceania (approx)
        };

        // Fill table (start at row 1)
        for (int i = 0; i < continents.Length; i++)
        {
            int row = i + 1;
            worksheet.Cells[row, 0].Value = continents[i];
            worksheet.Cells[row, 1].Value = areas[i];
        }

        // Format the Area column with thousands separators and align right
        worksheet.Columns[1].Style.NumberFormat = "#,##0";
        worksheet.Columns[1].Style.HorizontalAlignment = HorizontalAlignmentStyle.Right;

        // Autofit columns
        worksheet.Columns[0].AutoFit();
        worksheet.Columns[1].AutoFit();

        // Add a Pie chart to the right of the table (column index 3)
        // Parameters: (chartType, leftColumn, topRow, widthInPixels, heightInPixels)
        var chart = worksheet.Charts.Add(ChartType.Pie, 3, 0, 480, 320);

        // Select data range for chart: include header and data rows
        // GetSubrangeAbsolute(startRow, startColumn, endRow, endColumn)
        int lastDataRow = continents.Length; // header is row 0, data rows 1..continents.Length
        var dataRange = worksheet.Cells.GetSubrangeAbsolute(0, 0, lastDataRow, 1);

        // The 'true' means first column in range is category labels (continents)
        chart.SelectData(dataRange, true);

        // Chart title
        chart.Title.Text = "Landmass breakdown";
        chart.Title.IsVisible = true;

        // Format the single pie series data labels to show category name, value and percentage
        var series = chart.Series[0];
        series.DataLabels.LabelContainsCategoryName = true;
        series.DataLabels.LabelContainsValue = true;
        series.DataLabels.LabelContainsPercentage = true;
        // Use thousands separator for the numeric value shown in the labels
        series.DataLabels.NumberFormat = "#,##0";
        // Separator between the parts (category, value, percentage)
        series.DataLabels.Separator = " - ";
        // Position labels outside the slices
        series.DataLabels.LabelPosition = DataLabelPosition.OutsideEnd;
        series.DataLabels.ShowLeaderLines = true;

        // Optionally hide legend (pie labels already show category names)
        chart.Legend.IsVisible = false;

        // Save workbook
        workbook.Save(fileName);

        Console.WriteLine($"Saved: {fileName}");
    }
}