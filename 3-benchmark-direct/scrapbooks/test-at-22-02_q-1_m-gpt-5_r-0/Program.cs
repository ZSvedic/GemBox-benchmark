using System;
using GemBox.Spreadsheet;
using GemBox.Spreadsheet.Charts;

class Program
{
    static void Main()
    {
        // If using the Professional version, put your serial key below.
        SpreadsheetInfo.SetLicense("FREE-LIMITED-KEY");

        // Create workbook and a "Breakdown" sheet.
        var workbook = new ExcelFile();
        var ws = workbook.Worksheets.Add("Breakdown");

        // Header.
        ws.Cells[0, 0].Value = "Continents";
        ws.Cells[0, 1].Value = "Area (km2)";
        ws.Rows[0].Style.Font.Weight = ExcelFont.BoldWeight; // make header bold

        // Data (7-continent model; approximate land areas in km²).
        var data = new (string Name, int Area)[]
        {
            ("Africa",        30370000),
            ("Antarctica",    14000000),
            ("Asia",          44579000),
            ("Europe",        10180000),
            ("North America", 24709000),
            ("South America", 17840000),
            ("Oceania",        8600000),
        };

        for (int i = 0; i < data.Length; i++)
        {
            ws.Cells[i + 1, 0].Value = data[i].Name;
            ws.Cells[i + 1, 1].SetValue(data[i].Area);
        }

        // Use thousands separators for km2 values.
        ws.Columns[1].Style.NumberFormat = "#,##0";

        // Autofit both columns.
        ws.Columns[0].AutoFit();
        ws.Columns[1].AutoFit();

        // Create a pie chart to the right of the table.
        // Anchor next to columns A:B (data range A1:B8).
        var pie = ws.Charts.Add<PieChart>("D2", "L25");
        pie.SelectData(ws.Cells.GetSubrange("A1:B8"), true);

        // Title.
        pie.Title.Text = "Landmass breakdown";

        // Data labels: show continent name, area, and percentage, with leader lines.
        var labels = pie.Series[0].DataLabels;
        labels.Show(DataLabelPosition.OutsideEnd);
        labels.LabelContainsCategoryName = true;   // continent name
        labels.LabelContainsValue = true;          // area
        labels.NumberFormat = "#,##0";             // area format with thousands separators
        labels.LabelContainsPercentage = true;     // percentage
        labels.Separator = "\n";                   // put each part on its own line
        labels.ShowLeaderLines = true;

        // Save as "Earth–HHhMMm.xlsx" using current 24-hour time.
        var now = DateTime.Now;
        var fileName = $"Earth–{now:HH'h'mm'm'}.xlsx";
        workbook.Save(fileName);
    }
}