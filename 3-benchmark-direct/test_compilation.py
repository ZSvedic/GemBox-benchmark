# Python stdlib.
import asyncio
import dataclasses as dc

# Third-party.
import dotenv

# Local modules.
import base_classes as bc
import metrics as mt
import questions as qs
import benchmark
from tee_logging import logging_context

# Main test.

_PROMPT = '''You are GemBoxGPT, a GPT chatbot that provides coding assistance for GemBox.Spreadsheet. 
Assist only with GemBox-related queries. When giving code, give the complete C# code marked with ```csharp ... ``` blocks.
If you don't know something, first open page of the official GemBox.Spreadsheet example, as examples provide the most dense information. Below is the list of all official examples.
If after reading examples you don't understand class/method/property then search for official API documentation. Limit documentation search to "site:https://www.gemboxsoftware.com/spreadsheet/docs".
In total, do not make more than 3 search queries and browse more than 5 pages—GemBox website is small and your time to first token is 50 sec.

BASE: https://www.gemboxsoftware.com/spreadsheet/examples/
EXAMPLES:
asp-net-core-create-excel/5601
asp-net-excel-export-gridview/5101
asp-net-excel-viewer/6012
blazor-create-excel/5602
c-sharp-convert-excel-to-image/405
c-sharp-convert-excel-to-pdf/404
c-sharp-create-write-excel-file/402
c-sharp-excel-range/204
c-sharp-export-datatable-dataset-to-excel/501
c-sharp-export-excel-to-datatable/502
c-sharp-microsoft-office-interop-excel-automation/6004
c-sharp-open-read-excel-file/401
c-sharp-read-write-csv/122
c-sharp-vb-net-convert-excel-html/117
c-sharp-vb-net-create-excel-chart-sheet/302
c-sharp-vb-net-create-excel-tables/119
c-sharp-vb-net-excel-chart-formatting/306
c-sharp-vb-net-excel-conditional-formatting/105
c-sharp-vb-net-excel-form-controls/123
c-sharp-vb-net-excel-row-column-autofit/108
c-sharp-vb-net-excel-style-formatting/202
c-sharp-vb-net-import-export-excel-datagridview/5301
c-sharp-vb-net-print-excel/451
c-sharp-vb-net-xls-decryption/707
convert-excel-table-range-vb-net-c-sharp/6011
convert-import-write-json-to-excel-vb-net-c-sharp/6010
convert-xls-xlsx-ods-csv-html-net-c-sharp/6001
create-excel-charts/301
create-excel-file-maui/5802
create-excel-file-xamarin/5801
create-excel-files-c-sharp/6013
create-excel-pdf-on-azure/5901
create-excel-pdf-on-docker-net-core/5902
create-excel-pdf-on-linux-net-core/5701
create-excel-pivot-tables/114
create-read-write-excel-classic-asp/5501
create-read-write-excel-php/5502
create-read-write-excel-python/5503
edit-save-excel-template/403
excel-autofilter/112
excel-calculations-c-sharp-vb-net/6022
excel-cell-comments/208
excel-cell-data-types/201
excel-cell-hyperlinks/207
excel-cell-inline-formatting/203
excel-cell-number-format/205
excel-chart-components/304
excel-charts-guide-c-sharp/6019
excel-data-validation/106
excel-defined-names/214
excel-encryption/701
excel-find-replace-text/109
excel-formulas/206
excel-freeze-split-panes/102
excel-grouping/101
excel-header-footer-formatting/6021
excel-headers-footers/210
excel-images/209
excel-performance-metrics/5401
excel-preservation/801
excel-print-title-area/104
excel-print-view-options/103
excel-properties/107
excel-shapes/211
excel-sheet-copy-delete/111
excel-sheet-protection/704
excel-textboxes/212
excel-workbook-protection/705
excel-wpf/5201
excel-xlsx-digital-signature/706
fixed-columns-width-text/118
fonts/115
free-trial-professional/1001
getting-started/601
merge-excel-cells-c-sharp-vb-net/213
open-read-excel-files-c-sharp/6009
pdf-digital-signature/703
pdf-encryption/702
progress-reporting-and-cancellation/121
protecting-excel-data-c-sharp/6020
right-to-left-text/120
sort-data-excel/113
unit-conversion-excel/116
vba-macros/124
xlsx-write-protection/708
'''

_TASK_HELLO = '''Write C# code that uses GemBox.Spreadsheet to create an Excel file with 'Hello!' in cell A1.'''

_TASK_CHART = '''Using GemBox.Spreadsheet, generate C# code to create "Earth–HHhMMm.xlsx", where HH and MM are current hours and minutes (24-hour time). 
A "Breakdown" sheet has columns "Continents" and "Area (km2)". List all known continents and respective areas that you know. Use thousands separators for km2, make columns autofit, and make header bold. 
Right to the table, create a pie chart named "Landmass breakdown" that shows continent's area percentage. Each pie should have a label with continent name, area, and percentage. '''

_QUESTIONS = [
    # qs.QuestionData( category='compilation', question=_TASK_HELLO, masked_code='', answers=[] ),
    qs.QuestionData( category='compilation', question=_TASK_CHART, masked_code='', answers=[] ),
]

async def main_test():
    print("\n===== test_compilation.main_test() =====")

    # Filter models.
    models = (
        bc.Models()
        .by_names(['gemini-2.5-flash']) # 'google/gemini-3-pro-preview', 'gpt-5-mini', 'gpt-5'
    )
    print(f"Filtered models ({len(models)}): {models}")

    # Starting context.
    s_ctx = benchmark.BenchmarkContext(
        models=models,
        verbose=True, 
        system_ins=_PROMPT,
        questions=_QUESTIONS,
        parse_type=None,
        bench_n_times=1, 
    )
    
    # Testing contexts.
    contexts = [
        # dc.replace(s_ctx, description='Plain call + low reasoning', 
        #            reasoning='low', timeout_sec=40),
        # dc.replace(s_ctx, description='Plain call + medium reasoning',
        #            reasoning='medium', timeout_sec=60),   
        dc.replace(s_ctx, description='Web search (domain) + low reasoning',
                   reasoning='low', web=True, include_domains='gemboxsoftware.com', timeout_sec=180),
    ]
    
    # Benchmark models.
    perf_data = [await benchmark.benchmark_context(ctx) for ctx in contexts]

    # Print summary.
    print(f"\n=== SUMMARY OF ALL TESTS in test_prompts.py ===")
    mt.print_metrics(perf_data, True)

if __name__ == "__main__":
    # Load environment variables from parent directory .env.
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")
    
    with logging_context("test_compilation"):
        asyncio.run(main_test())