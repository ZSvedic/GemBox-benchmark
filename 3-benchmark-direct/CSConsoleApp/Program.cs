using System;
using GemBox.Spreadsheet;

class Program
{
    static void Main()
    {
        // If you have a license key, put it here:
        // SpreadsheetInfo.SetLicense("YOUR-LICENSE-KEY");
        // For the free version, use the free key:
        SpreadsheetInfo.SetLicense("FREE-LIMITED-KEY");

        // Create a new Excel file and worksheet.
        var workbook = new ExcelFile();
        var worksheet = workbook.Worksheets.Add("Sheet1");

        // Put "Hello!" into cell A1.
        worksheet.Cells["A1"].Value = "Hello!";

        // Save the file to the current folder.
        workbook.Save("Hello.xlsx");

        Console.WriteLine("Saved Hello.xlsx");
    }
}