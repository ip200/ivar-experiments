
import numpy as np
import pandas as pd


def GetDataset(name, base_path):
    """ Load a dataset
    
    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"
    
    Returns
    -------
    X : features (nXp)
    y : labels (n)
    
	"""

    if name=="star":
        df = pd.read_csv(base_path + 'STAR.csv').astype(object)
        df.loc[df['gender'] == 'female', 'gender'] = 0
        df.loc[df['gender'] == 'male', 'gender'] = 1
        
        df.loc[df['ethnicity'] == 'cauc', 'ethnicity'] = 0
        df.loc[df['ethnicity'] == 'afam', 'ethnicity'] = 1
        df.loc[df['ethnicity'] == 'asian', 'ethnicity'] = 2
        df.loc[df['ethnicity'] == 'hispanic', 'ethnicity'] = 3
        df.loc[df['ethnicity'] == 'amindian', 'ethnicity'] = 4
        df.loc[df['ethnicity'] == 'other', 'ethnicity'] = 5
        
        df.loc[df['stark'] == 'regular', 'stark'] = 0
        df.loc[df['stark'] == 'small', 'stark'] = 1
        df.loc[df['stark'] == 'regular+aide', 'stark'] = 2
        
        df.loc[df['star1'] == 'regular', 'star1'] = 0
        df.loc[df['star1'] == 'small', 'star1'] = 1
        df.loc[df['star1'] == 'regular+aide', 'star1'] = 2        
        
        df.loc[df['star2'] == 'regular', 'star2'] = 0
        df.loc[df['star2'] == 'small', 'star2'] = 1
        df.loc[df['star2'] == 'regular+aide', 'star2'] = 2   

        df.loc[df['star3'] == 'regular', 'star3'] = 0
        df.loc[df['star3'] == 'small', 'star3'] = 1
        df.loc[df['star3'] == 'regular+aide', 'star3'] = 2      
        
        df.loc[df['lunchk'] == 'free', 'lunchk'] = 0
        df.loc[df['lunchk'] == 'non-free', 'lunchk'] = 1
        
        df.loc[df['lunch1'] == 'free', 'lunch1'] = 0    
        df.loc[df['lunch1'] == 'non-free', 'lunch1'] = 1      
        
        df.loc[df['lunch2'] == 'free', 'lunch2'] = 0    
        df.loc[df['lunch2'] == 'non-free', 'lunch2'] = 1  
        
        df.loc[df['lunch3'] == 'free', 'lunch3'] = 0    
        df.loc[df['lunch3'] == 'non-free', 'lunch3'] = 1  
        
        df.loc[df['schoolk'] == 'inner-city', 'schoolk'] = 0
        df.loc[df['schoolk'] == 'suburban', 'schoolk'] = 1
        df.loc[df['schoolk'] == 'rural', 'schoolk'] = 2  
        df.loc[df['schoolk'] == 'urban', 'schoolk'] = 3

        df.loc[df['school1'] == 'inner-city', 'school1'] = 0
        df.loc[df['school1'] == 'suburban', 'school1'] = 1
        df.loc[df['school1'] == 'rural', 'school1'] = 2  
        df.loc[df['school1'] == 'urban', 'school1'] = 3      
        
        df.loc[df['school2'] == 'inner-city', 'school2'] = 0
        df.loc[df['school2'] == 'suburban', 'school2'] = 1
        df.loc[df['school2'] == 'rural', 'school2'] = 2  
        df.loc[df['school2'] == 'urban', 'school2'] = 3      
        
        df.loc[df['school3'] == 'inner-city', 'school3'] = 0
        df.loc[df['school3'] == 'suburban', 'school3'] = 1
        df.loc[df['school3'] == 'rural', 'school3'] = 2  
        df.loc[df['school3'] == 'urban', 'school3'] = 3  
        
        df.loc[df['degreek'] == 'bachelor', 'degreek'] = 0
        df.loc[df['degreek'] == 'master', 'degreek'] = 1
        df.loc[df['degreek'] == 'specialist', 'degreek'] = 2  
        df.loc[df['degreek'] == 'master+', 'degreek'] = 3 

        df.loc[df['degree1'] == 'bachelor', 'degree1'] = 0
        df.loc[df['degree1'] == 'master', 'degree1'] = 1
        df.loc[df['degree1'] == 'specialist', 'degree1'] = 2  
        df.loc[df['degree1'] == 'phd', 'degree1'] = 3              
        
        df.loc[df['degree2'] == 'bachelor', 'degree2'] = 0
        df.loc[df['degree2'] == 'master', 'degree2'] = 1
        df.loc[df['degree2'] == 'specialist', 'degree2'] = 2  
        df.loc[df['degree2'] == 'phd', 'degree2'] = 3
        
        df.loc[df['degree3'] == 'bachelor', 'degree3'] = 0
        df.loc[df['degree3'] == 'master', 'degree3'] = 1
        df.loc[df['degree3'] == 'specialist', 'degree3'] = 2  
        df.loc[df['degree3'] == 'phd', 'degree3'] = 3          
        
        df.loc[df['ladderk'] == 'level1', 'ladderk'] = 0
        df.loc[df['ladderk'] == 'level2', 'ladderk'] = 1
        df.loc[df['ladderk'] == 'level3', 'ladderk'] = 2  
        df.loc[df['ladderk'] == 'apprentice', 'ladderk'] = 3  
        df.loc[df['ladderk'] == 'probation', 'ladderk'] = 4
        df.loc[df['ladderk'] == 'pending', 'ladderk'] = 5
        df.loc[df['ladderk'] == 'notladder', 'ladderk'] = 6
        
        
        df.loc[df['ladder1'] == 'level1', 'ladder1'] = 0
        df.loc[df['ladder1'] == 'level2', 'ladder1'] = 1
        df.loc[df['ladder1'] == 'level3', 'ladder1'] = 2  
        df.loc[df['ladder1'] == 'apprentice', 'ladder1'] = 3  
        df.loc[df['ladder1'] == 'probation', 'ladder1'] = 4
        df.loc[df['ladder1'] == 'noladder', 'ladder1'] = 5
        df.loc[df['ladder1'] == 'notladder', 'ladder1'] = 6
        
        df.loc[df['ladder2'] == 'level1', 'ladder2'] = 0
        df.loc[df['ladder2'] == 'level2', 'ladder2'] = 1
        df.loc[df['ladder2'] == 'level3', 'ladder2'] = 2  
        df.loc[df['ladder2'] == 'apprentice', 'ladder2'] = 3  
        df.loc[df['ladder2'] == 'probation', 'ladder2'] = 4
        df.loc[df['ladder2'] == 'noladder', 'ladder2'] = 5
        df.loc[df['ladder2'] == 'notladder', 'ladder2'] = 6
        
        df.loc[df['ladder3'] == 'level1', 'ladder3'] = 0
        df.loc[df['ladder3'] == 'level2', 'ladder3'] = 1
        df.loc[df['ladder3'] == 'level3', 'ladder3'] = 2  
        df.loc[df['ladder3'] == 'apprentice', 'ladder3'] = 3  
        df.loc[df['ladder3'] == 'probation', 'ladder3'] = 4
        df.loc[df['ladder3'] == 'noladder', 'ladder3'] = 5
        df.loc[df['ladder3'] == 'notladder', 'ladder3'] = 6
        
        df.loc[df['tethnicityk'] == 'cauc', 'tethnicityk'] = 0
        df.loc[df['tethnicityk'] == 'afam', 'tethnicityk'] = 1
        
        df.loc[df['tethnicity1'] == 'cauc', 'tethnicity1'] = 0
        df.loc[df['tethnicity1'] == 'afam', 'tethnicity1'] = 1
        
        df.loc[df['tethnicity2'] == 'cauc', 'tethnicity2'] = 0
        df.loc[df['tethnicity2'] == 'afam', 'tethnicity2'] = 1
        
        df.loc[df['tethnicity3'] == 'cauc', 'tethnicity3'] = 0
        df.loc[df['tethnicity3'] == 'afam', 'tethnicity3'] = 1
        df.loc[df['tethnicity3'] == 'asian', 'tethnicity3'] = 2
        
        df = df.dropna()
        
        grade = df["readk"] + df["read1"] + df["read2"] + df["read3"]
        grade += df["mathk"] + df["math1"] + df["math2"] + df["math3"]
        
        
        names = df.columns
        target_names = names[8:16]
        data_names = np.concatenate((names[0:8],names[17:]))
        X = df.loc[:, data_names].values
        y = grade.values


        
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    return X, y
        