using GemBox.Spreadsheet;
using GemBox.Spreadsheet.Charts;

public static class Charts
{
    public static void Run()
    {
        var workbook = new ExcelFile();
        var worksheet = workbook.Worksheets.Add("Chart");

        worksheet.Cells["A1"].Value = "Name";
        worksheet.Cells["A2"].Value = "John Doe";
        worksheet.Cells["A3"].Value = "Fred Nurk";
        worksheet.Cells["A4"].Value = "Hans Meier";
        worksheet.Cells["A5"].Value = "Ivan Horvat";
        worksheet.Cells["B1"].Value = "Salary";
        worksheet.Cells["B2"].Value = 3600;
        worksheet.Cells["B3"].Value = 2580;
        worksheet.Cells["B4"].Value = 3200;
        worksheet.Cells["B5"].Value = 4100;

        // Question: How do you add a column chart to a worksheet between cells D2 and M25?
        // Mask: \bCharts.Add\b
        // Mask: \bChartType\.Column\b
        // Mask: "D2"
        // Mask: "M25"
        var chart=worksheet.Charts.Add(ChartType.Column,"D2","M25");

        // Question: How do you set the chart's data range to (0,0,4,1), including headers?
        // Mask: \bSelectData\b
        // Mask: \bCells.GetSubrangeAbsolute\b
        // Mask: \btrue\b
        chart.SelectData(worksheet.Cells.GetSubrangeAbsolute(0,0,4,1),true);

        workbook.Save(Path.Combine(Program.WritePath,"Charts.xlsx"));
    }
} 