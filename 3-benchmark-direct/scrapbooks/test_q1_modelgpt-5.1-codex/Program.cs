using System;
using GemBox.Spreadsheet;
using GemBox.Spreadsheet.Charts;

class Program
{
    static void Main()
    {
        SpreadsheetInfo.SetLicense("FREE-LIMITED-KEY");

        string timestamp = DateTime.Now.ToString("HH'h'mm'm'");
        string fileName = $"Earth–{timestamp}.xlsx";

        var workbook = new ExcelFile();
        var worksheet = workbook.Worksheets.Add("Breakdown");

        // Header.
        worksheet.Cells["A1"].Value = "Continents";
        worksheet.Cells["B1"].Value = "Area (km2)";
        worksheet.Cells.GetSubrange("A1:B1").Style.Font.Weight = ExcelFont.BoldWeight;

        var continents = new (string Name, double Area)[]
        {
            ("Africa",        30370000),
            ("Antarctica",    14000000),
            ("Asia",          44579000),
            ("Europe",        10180000),
            ("North America", 24709000),
            ("Oceania",        8526000),
            ("South America", 17840000)
        };

        // Populate rows.
        for (int i = 0; i < continents.Length; i++)
        {
            int row = i + 2;
            worksheet.Cells[row, 0].Value = continents[i].Name;
            worksheet.Cells[row, 1].Value = continents[i].Area;
        }

        // Format Area column.
        worksheet.Cells.GetSubrange($"B2:B{continents.Length + 1}").Style.NumberFormat = "#,##0";

        // Autofit columns.
        worksheet.Columns[0].AutoFit();
        worksheet.Columns[1].AutoFit();

        // Pie chart to the right of the table.
        var chart = worksheet.Charts.Add(ChartType.Pie, "D2", "K18");
        chart.Title.Text = "Landmass breakdown";

        var dataRange = worksheet.Cells.GetSubrangeAbsolute(0, 0, continents.Length, 1); // Includes header.
        chart.SelectData(dataRange, true);

        chart.Legend.Position = ChartLegendPosition.Right;
        chart.DataLabels.LabelPosition = DataLabelPosition.BestFit;
        chart.DataLabels.ShowCategoryName = true;
        chart.DataLabels.ShowValue = true;
        chart.DataLabels.NumberFormat = "#,##0\" km²\"";
        chart.DataLabels.ShowPercentage = true;
        chart.DataLabels.Separator = "\n";

        workbook.Save(fileName);
    }
}