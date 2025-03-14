class Dataloader:

    def __init__(self):
        print("Initializing Dataloader")
        self.df        = None
        self.nCols     = 0
        self.df_train  = None
        self.df_test   = None
        self.label_col = None


    def load(self, filename, label_col=None):
        
        df = pd.read_csv(filename)
        self.nCols = len(df.columns)

        if(label_col is not None):
            self.set_label(label_col)

        return df


    def clean_data(self, df):

        df = df.dropna()
        df = df.drop_duplicates()

        return df


    def set_label(self, label_col):

        self.label_col = label_col


    def column_transform(self, df, col_idx, transform_type):

        if(transform_type == 'scale'):
            df[col_idx] = df[col_idx] / scale_factor
        elif(transform_type == 'normalize'):
            df[col_idx] = (df[col_idx] - df[col_idx].min()) / (df[col_idx].max() - df[col_idx].min())



    def scale_data(self, df, scale_factor=None):

        if scale_factor is None:
            scale_factor = [1.0, 1.0, 1.0]
        elif(len(scale_factor) != self.nCols):
            raise ValueError("scale_factor must be a list of length nCols")

        # Create a copy to avoid modifying the original DataFrame
        scaled_df = df.copy()
        
        # Scale each column by its corresponding scale factor
        for col, factor in zip(df.columns, scale_factor):
            scaled_df[col] = df[col] / factor


    def train_test_split(self, df, test_size=0.2):

        n = len(df)
        n_test = int(n * test_size)
        n_train = n - n_test

        df_train = df.iloc[:n_train]
        df_test = df.iloc[n_train:]

        self.df_train = df_train
        self.df_test = df_test



    def break_into_batches(self, df, batch_size):

        n = len(df)
        n_batches = n // batch_size

        batches = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch = df.iloc[start_idx:end_idx]
            batches.append(batch)
            
        # Handle remaining data if n is not perfectly divisible by batch_size
        if n % batch_size != 0:
            last_batch = df.iloc[n_batches * batch_size:]
            batches.append(last_batch)

        return batches
        


    