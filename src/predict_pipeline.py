import os
import yaml
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import pickle
from datetime import datetime
from src.utility_file import Utility

class Predict:
    def __init__(self) -> None:
        pass

    def predict_pipeline(self):

        info='ocean twelve raid box office ocean twelve crime caper sequel starring george clooney brad pitt julia robert gone straight number one u box office chart took 40 8m 21m weekend ticket sale according studio estimate sequel follows master criminal try pull three major heist across europe knocked last week number one national treasure third place wesley snipe blade trinity second taking 16 1m 8 4m rounding top five animated fable polar express starring tom hank festive comedy christmas kranks ocean twelve box office triumph mark fourth biggest opening december release u three film lord ring trilogy sequel narrowly beat 2001 predecessor ocean eleven took 38 1m 19 8m opening weekend 184m 95 8m total remake 1960s film starring frank sinatra rat pack ocean eleven directed oscar winning director steven soderbergh soderbergh return direct hit sequel reunites clooney pitt robert matt damon andy garcia elliott gould catherine zeta jones join star cast fun good holiday movie said dan fellman president distribution warner bros however u critic le complimentary 110m 57 2m project los angeles time labelling dispiriting vanity project milder review new york time dubbed sequel unabashedly trivial'
        headlines=[Utility.clean_text(info)]
        print(headlines)


        cv=joblib.load('models\count_vector.joblib')
        print(cv)
        model= joblib.load('models\grid_search_model.joblib')
        

        y_pred1 = cv.transform(headlines)
        print(y_pred1)
        pred=model.predict(y_pred1)[0]
        # prediction=y_pred1.toarray()
        
        print('should be: entertainment')
        print('result: ',pred)
        # return pred


    