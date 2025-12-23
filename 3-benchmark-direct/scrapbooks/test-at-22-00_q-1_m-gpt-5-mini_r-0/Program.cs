using System;
using GemBox.Spreadsheet;
using GemBox.Spreadsheet.Charts;
using GemBox.Spreadsheet.Drawing;

class Program
{
    static void Main()
    {
        // If using the Professional version, put your serial key below.
        SpreadsheetInfo.SetLicense("FREE-LIMITED-KEY");

        // Create a new workbook and add the "Breakdown" worksheet.
        var workbook = new ExcelFile();
        var worksheet = workbook.Worksheets.Add("Breakdown");

        // Header
        worksheet.Cells["A1"].Value = "Continents";
        worksheet.Cells["B1"].Value = "Area (km2)";

        // Make header bold.
        worksheet.Rows[0].Style.Font.Weight = ExcelFont.BoldWeight;

        // Continent names and approximate areas (km^2).
        var continents = new (string Name, double AreaKm2)[]
        {
            ("Asia", 44579000),         // ~44,579,000 km2
            ("Africa", 30370000),       // ~30,370,000 km2
            ("North America", 24709000),// ~24,709,000 km2
            ("South America", 17840000),// ~17,840,000 km2
            ("Antarctica", 14000000),   // ~14,000,000 km2
            ("Europe", 10180000),       // ~10,180,000 km2
            ("Australia/Oceania", 8600000) // ~8,600,000 km2
        };

        // Write data to sheet starting at row 2 (index 1).
        for (int i = 0; i < continents.Length; i++)
        {
            int row = i + 1; // row index (0-based)
            worksheet.Cells[row, 0].Value = continents[i].Name;
            worksheet.Cells[row, 1].Value = continents[i].AreaKm2;
        }

        // Format the area column with thousands separators (no decimals).
        worksheet.Columns[1].Style.NumberFormat = "#,##0";

        // Autofit the two columns so content is visible.
        worksheet.Columns[0].AutoFit();
        worksheet.Columns[1].AutoFit();

        // Create a pie chart positioned to the right of the table.
        // Chart covers cells D2:K20 (adjust size/position as you prefer).
        var chart = worksheet.Charts.Add<PieChart>("D2", "K20");

        // Select data for the chart (A1:B8 includes header row + 7 data rows).
        // Pass 'true' to indicate first row contains series/category names.
        chart.SelectData(worksheet.Cells.GetSubrangeAbsolute(0, 0, continents.Length, 1), true);

        // Set chart title.
        chart.Title.Text = "Landmass breakdown";
        chart.Title.TextFormat.Size = Length.From(14, LengthUnit.Point);

        // Configure data labels for the series to show category name, value and percentage.
        var series = chart.Series[0];
        series.DataLabels.LabelContainsCategoryName = true;  // continent name
        series.DataLabels.LabelContainsValue = true;         // numeric area
        series.DataLabels.LabelContainsPercentage = true;    // percent of total
        series.DataLabels.NumberFormat = "#,##0";            // formats the numeric value in label
        series.DataLabels.Separator = "\n";                  // put each part on its own line
        series.DataLabels.Show(DataLabelPosition.BestFit);
        series.DataLabels.ShowLeaderLines = true;

        // Optional: hide legend if desired, or place it at the right.
        chart.Legend.IsVisible = true;
        chart.Legend.Position = ChartLegendPosition.Right;

        // Build filename with en dash and current time (24-hour HH and mm).
        var now = DateTime.Now;
        string filename = $"Earth\u2013{now:HH}h{now:mm}m.xlsx";

        // Save workbook.
        workbook.Save(filename);

        Console.WriteLine($"Created \"{filename}\" with a \"Breakdown\" sheet and pie chart.");
    }
}