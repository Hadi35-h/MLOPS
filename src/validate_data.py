from pathlib import Path
import sys
import great_expectations as gx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"


def run_data_validation(file_path: Path):
    if not file_path.exists():
        print(f"❌ لم يتم العثور على ملف البيانات في المسار: {file_path}")
        sys.exit(1)

    # 1. قراءة البيانات مع تنظيف أسماء الأعمدة وإزالة الأسطر الفارغة
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # إزالة أي مساحات مخفية في أسماء الأعمدة

    # 2. إنشاء سياق Great Expectations
    context = gx.get_context(mode="ephemeral")

    # 3. إعداد مصدر البيانات
    data_source = context.data_sources.add_pandas("pandas_datasource")
    data_asset = data_source.add_dataframe_asset(name="pandas_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "batch_definition"
    )

    # 4. بناء قائمة التوقعات الفعالة بمرونة
    suite = context.suites.add(gx.ExpectationSuite(name="data_quality_suite"))

    # التأكد من وجود أعمدة رئيسية
    for col in ["order_id", "product_id", "price"]:
        if col in df.columns:
            suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    # 5. تشغيل الفحص
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="validation_definition",
            data=batch_definition,
            suite=suite,
        )
    )

    results = validation_definition.run(batch_parameters={"dataframe": df})

    if not results.success:
        print("\n❌ فشل فحص جودة البيانات! تفاصيل الاختبارات الفاشلة:")
        for res in results.results:
            if not res.success:
                expectation_type = res.expectation_config.type
                kwargs = res.expectation_config.kwargs
                print(f" - الاختبار الفاشل: {expectation_type} على المعاملات {kwargs}")
        sys.exit(1)

    print("✅ تم فحص جودة البيانات بنجاح: جميع الشروط مطابقة!")


if __name__ == "__main__":
    run_data_validation(DATA_PATH)
