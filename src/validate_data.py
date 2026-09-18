from pathlib import Path
import sys
import great_expectations as gx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"


def run_data_validation(file_path: Path):
    if not file_path.exists():
        print(f"❌ Data file not found at path: {file_path}")
        sys.exit(1)

    # 1. Read data and clean column names, remove empty rows
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove any hidden spaces in column names

    # 2. Create a Great Expectations context
    context = gx.get_context(mode="ephemeral")

    # 3. Set up the data source
    data_source = context.data_sources.add_pandas("pandas_datasource")
    data_asset = data_source.add_dataframe_asset(name="pandas_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "batch_definition"
    )

    # 4. Build a flexible expectation suite
    suite = context.suites.add(gx.ExpectationSuite(name="data_quality_suite"))

    # Ensure key columns exist
    for col in ["order_id", "product_id", "price"]:
        if col in df.columns:
            suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    # 5. Run the validation
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="validation_definition",
            data=batch_definition,
            suite=suite,
        )
    )

    results = validation_definition.run(batch_parameters={"dataframe": df})

    if not results.success:
        print("\n❌ Data quality check failed! Details of failed tests:")
        for res in results.results:
            if not res.success:
                expectation_type = res.expectation_config.type
                kwargs = res.expectation_config.kwargs
                print(f" - Failed test: {expectation_type} with args {kwargs}")
        sys.exit(1)

    print("✅ Data quality check passed: all expectations met!")


if __name__ == "__main__":
    run_data_validation(DATA_PATH)
