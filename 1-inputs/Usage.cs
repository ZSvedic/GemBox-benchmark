using GemBox.Spreadsheet;

public static class Usage
{
    public static void Run()
    {
        // Question: How to set GemBox.Spreadsheet to use the free license?
        // Mask: \bSpreadsheetInfo\b
        // Mask: \bSetLicense\b
        SpreadsheetInfo.SetLicense("FREE-LIMITED-KEY");
    }
} 