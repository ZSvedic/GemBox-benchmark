using System;
using GemBox.Spreadsheet;

class Program
{
    static void Main()
    {
        // If you have a GemBox.Spreadsheet Professional license key, uncomment the following line
        // SpreadsheetInfo.SetLicense("YOUR-LICENSE-KEY");

        // If using Free mode, no license key is required. Note that Free mode has certain limitations.
        // Free mode supports charts with up to 150 rows of data.

        var workbook = new ExcelFile();
        var worksheet = workbook.Worksheets.Add("Breakdown");

        // Headers
        worksheet.Cells["A1"].Value = "Continents";
        worksheet.Cells["B1"].Value = "Area (km2)";
        worksheet.Cells["A1:B1"].Style.Font.Bold = true;

        // Continent data
        string[] continents = {
            "Asia", "Africa", "North America", "South America",
            "Antarctica", "Europe", "Australia"
        };
        double[] areas = {
            44579000, 30370000, 24709000, 17840000,
            14200000, 10180000, 7692000
        };

        // Fill data
        for (int i = 0; i < continents.Length; i++)
        {
            worksheet.Cells[ExcelCellAddress.FromRowColumnIndex(i + 2, 0)].Value = continents[i]; // Column A
            worksheet.Cells[ExcelCellAddress.FromRowColumnIndex(i + 2, 1)].Value = areas[i];     // Column B
        }

        // Format Area column with thousands separator
        worksheet.Cells.GetSubrangeAbsolute("B2:B8").Style.NumberFormat = new ExcelNumberFormat("#,##0");

        // Autofit columns
        worksheet.Columns.AutoFit(1.0);

        // Add pie chart to the right of the table (positioned at D2, sized to K20)
        var pieChart = worksheet.Charts.Add(ExcelChartType.Pie, "D2", "K20");
        pieChart.Name = "Landmass breakdown";

        // Configure the pie chart series
        var series = pieChart.Series.Add("Landmass");
        series.Values = worksheet.Cells.GetSubrangeAbsolute("B2:B8");
        series.Categories = worksheet.Cells.GetSubrangeAbsolute("A2:A8");

        // Configure data labels to show continent name, area value, and percentage
        series.DataLabels.ShowCategoryName = true;
        series.DataLabels.ShowValue = true;
        series.DataLabels.ShowPercentage = true;
        series.DataLabels.Separator = " | "; // Custom separator for readability: "Continent | 44,579,000 | 29.93%"

        // Save the file with current time in filename (24-hour format)
        string currentHour = DateTime.Now.ToString("HH");
        string currentMinute = DateTime.Now.ToString("mm");
        string filename = $"../Earth–{currentHour}h{currentMinute}m.xlsx";
        workbook.Save(filename);

        Console.WriteLine($"Excel file saved as: {filename}");
    }
}