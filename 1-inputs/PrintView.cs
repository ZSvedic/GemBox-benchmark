using GemBox.Spreadsheet;

public static class PrintView
{
    public static void Run()
    {
        var workbook = new ExcelFile();
        var worksheet = workbook.Worksheets.Add("Sheet1");

        worksheet.Cells["M1"].Value = "This worksheet shows how to set various print related and view related options.";
        worksheet.Cells["M2"].Value = "To see results of print options, go to Print and Page Setup dialogs in MS Excel.";
        worksheet.Cells["M3"].Value = "Notice that print and view options are worksheet based, not workbook based.";

        // Question: How do you access the printing options of the "Sheet1" worksheet?
        // Mask: \bWorksheets\b
        // Mask: \bPrintOptions\b
        var printingOpt = workbook.Worksheets["Sheet1"].PrintOptions;

        // Question: How do you enable printing of gridlines in the worksheet?
        // Mask: \bPrintOptions\.PrintGridlines\b
        // Mask: \btrue\b
        worksheet.PrintOptions.PrintGridlines = true;

        // Question: How do you enable printing of row and column headings?
        // Mask: \bPrintOptions\.PrintHeadings\b
        // Mask: \btrue\b
        worksheet.PrintOptions.PrintHeadings = true;

        // Question: How do you set the worksheet to print in landscape orientation?
        // Mask: \bPrintOptions\.Portrait\b
        // Mask: \bfalse\b
        worksheet.PrintOptions.Portrait = false;

        // Question: How do you set the paper to A3 for printing?
        // Mask: \bPrintOptions\.PaperType\b
        // Mask: \bPaperType\b
        worksheet.PrintOptions.PaperType = PaperType.A3;

        // Question: How do you set printing of five copies?
        // Mask: \bPrintOptions\.NumberOfCopies\b
        worksheet.PrintOptions.NumberOfCopies = 5;

        // Question: How do you set the first visible column in a worksheet to column D?
        // Mask: \bViewOptions\.FirstVisibleColumn\b
        // Mask: \b3\b
        worksheet.ViewOptions.FirstVisibleColumn = 3;

        // Question: How do you set the worksheet zoom level to 125%?
        // Mask: \bViewOptions\.Zoom\b
        // Mask: \b125\b
        worksheet.ViewOptions.Zoom = 125;

        // Question: How do you set the print area to a specific cell range (E1:U7)?
        // Mask: \bNamedRanges\.SetPrintArea\b
        // Mask: \bCells\.GetSubrange\b
        worksheet.NamedRanges.SetPrintArea(worksheet.Cells.GetSubrange("E1","U7"));

        workbook.Save(Path.Combine(Program.WritePath, "PrintView.xlsx"));
    }
} 