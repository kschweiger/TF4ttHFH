import pandas as pd


def mergeh5Files(listOfFiles, output):
    dataframes = [pd.read_hdf(_file) for _file in listOfFiles]

    mergedDF = pd.concat(dataframes)

    with pd.HDFStore(output, "a") as store:
        store.put("data", mergedDF, index=False)
