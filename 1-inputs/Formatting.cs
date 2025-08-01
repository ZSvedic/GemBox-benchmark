using GemBox.Spreadsheet;

public static class Formatting
{
    public static void Run()
    {
        var workbook = new ExcelFile();
        var worksheet = workbook.Worksheets.Add("Sheet1");

        worksheet.Cells["A1"].Value = "Bold Hello World!";
        // Question: How to set cell A1 text to bold?
        // Mask: \bStyle.Font.Weight\b
        // Mask: \bExcelFont.BoldWeight\b
        worksheet.Cells["A1"].Style.Font.Weight=ExcelFont.BoldWeight;
        
        worksheet.Cells["A4"].Value="Right Border";
        // Question: How do you set the right border of a cell?
        // Mask: \bStyle.Borders\[IndividualBorder.Right\]\.LineStyle\b
        worksheet.Cells["A4"].Style.Borders[IndividualBorder.Right].LineStyle=LineStyle.Thin;

        worksheet.Cells["A2"].Value=123;
        // Question: How do you format a cell to display a number with a unit (e.g., meters)?
        // Mask: \bStyle.NumberFormat\b
        worksheet.Cells["A2"].Style.NumberFormat="#\" m\"";

        worksheet.Cells["A3"].Value="Background Color";
        // Question: How do you set a cell's background color?
        // Mask: \bStyle.FillPattern.SetSolid\b
        // Mask: \bSpreadsheetColor\b
        worksheet.Cells["A3"].Style.FillPattern.SetSolid(SpreadsheetColor.FromArgb(221,235,247));

        worksheet.Cells.GetSubrange("A5","B5").Merged=true;
        worksheet.Cells["A5"].Value="Merged Range";
        // Question: How do you style a merged cell range to italic?
        // Mask: \bStyle.Font.Italic\b
        worksheet.Cells.GetSubrange("A5","B5").Style.Font.Italic=true;

        worksheet.Cells["A6"].Value="PartialFFonttSize";
        // Question: How do you set a font size for a part of a cell's text starting with the 9th character and ending with the 12th character, both inclusive?
        // Mask: \bGetCharacters\b
        // Mask: \b4\b
        // Mask: \bFont.Size\b
        worksheet.Cells["A6"].GetCharacters(8,4).Font.Size=18*20;

        worksheet.Cells["A7"].Value="PartialFFonttName";
        // Question: How do you set a font name for part of a cell's text starting with the 9th character and ending with the 12th character, both inclusive?
        // Mask: \bGetCharacters\b
        // Mask: \b8\b
        // Mask: \bFont.Name\b
        worksheet.Cells["A7"].GetCharacters(8,4).Font.Name="Comic Sans MS";

        // Question: How do you autofit the first column to fit its content?
        // Mask: \.AutoFit\b
        worksheet.Columns[0].AutoFit();

        // Question: How do you set the worksheet zoom level to 140%?
        // Mask: \bViewOptions.Zoom\b
        // Mask: \b140\b
        worksheet.ViewOptions.Zoom=140;
        
        workbook.Save(Path.Combine(Program.WritePath,"Formatting.xlsx"));
    }
} 