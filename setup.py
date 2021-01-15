from setuptools import setup, Extension

setup(
    ext_modules=[Extension('dedupe.cpredicates',
                           ['dedupe/cpredicates.pyx'])],
    long_description="""
    dedupe is a library that uses machine learning to perform de-duplication and entity resolution quickly on structured data. dedupe is the open source engine for `dedupe.io <https://dedupe.io>`_

    **dedupe** will help you:

    * **remove duplicate entries** from a spreadsheet of names and addresses
    * **link a list** with customer information to another with order history, even without unique customer id's
    * take a database of campaign contributions and **figure out which ones were made by the same person**, even if the names were entered slightly differently for each record

    dedupe takes in human training data and comes up with the best rules for your dataset to quickly and automatically find similar records, even with very large databases.
    """,  # noqa: E501
)
