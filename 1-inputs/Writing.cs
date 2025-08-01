using GemBox.Spreadsheet;

public static class Writing
{
    public static void Run()
    {
        var workbook = new ExcelFile();
        var worksheet = workbook.Worksheets.Add("Writing");

        // Question: How do you assign the text "Hello" to cell A1?
        // Mask: \.Value\b
        worksheet.Cells["A1"].Value="Hello";

        // Question: How do you assign the boolean value true to cell A2?
        // Mask: \.Value\b
        // Mask: \btrue\b
        worksheet.Cells["A2"].Value=true;

        // Question: How do you write a formula to a cell?
        // Mask: \.Formula\b
        worksheet.Cells["A5"].Formula="=SUM(A2:A4)";

        // Question: How do you set a cell value using row and column indices?
        // Mask: \b2\b
        // Mask: \b3\b
        worksheet.Cells[2,3].Value="Third row, fourth column";

        workbook.Save(Path.Combine(Program.WritePath,"Writing.xlsx"));
    }
} 