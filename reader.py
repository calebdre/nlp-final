class Reader:
    base_path = "data"
    data = {}
        
    def read(self):
        for lang in ["vi", "zh"]:
            self.data[lang] = {} 

            for d_set in ["train", "test", "dev"]:
                lang_path = "{}/iwslt-{}-en-processed".format(self.base_path, lang)
                data_path = "{}/{}.tok.{}".format(lang_path, d_set, lang)
                eng_path = "{}/{}.tok.en".format(lang_path, d_set)
                
                data = (open(data_path, "r").readlines(), open(eng_path, "r").readlines())
                self.data[lang][d_set] = data
                
    def get_train(self, lang):
        return self.data[lang]["train"]
   
    def get_validation(self, lang):
        return self.data[lang]["dev"]
   
    def get_test(self, lang):
        return self.data[lang]["test"]