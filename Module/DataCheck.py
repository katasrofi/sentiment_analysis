def check_data(dataset):

    # Null value
    null = dataset.isnull().sum()
    print(null)
    
    print("============================================")
    # Type data
    typedata = dataset.dtypes
    print(typedata)

    print("============================================")    
    # Shape
    shape = dataset.shape
    print(f"Shape of the data: ", shape)
    
    # Describe
    print("============================================")
    describe = dataset.describe()
    print(describe)
    
