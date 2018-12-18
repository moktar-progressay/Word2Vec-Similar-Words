import gensim
import cherrypy
import os
import pickle
import jsonpickle

if os.path.isfile('model.pik'):
    model = pickle.load(open("model.pik", "rb"))
else:
    model = gensim.models.KeyedVectors.load_word2vec_format('conceptnet.txt', binary=False)
    pickle.dump(model, open("model.pik", "wb"))

globalConfig = {
    'server.socket_host': '0.0.0.0',
    'server.socket_port': 8083,
    'tools.cors.on': True
}


class App:

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def find_similar_words(self, word, number_of_words):
        words = model.similar_by_word(word.lower(), topn=int(number_of_words))
        result = [list(w) for w in words]
        return jsonpickle.encode(result)


def cors():
    if cherrypy.request.method == 'OPTIONS':
        cherrypy.response.headers['Access-Control-Allow-Methods'] = 'POST'
        cherrypy.response.headers['Access-Control-Allow-Headers'] = 'content-type'
        cherrypy.response.headers['Access-Control-Allow-Origin'] = '*'
        # tell CherryPy no avoid normal handler
        return True
    else:
        cherrypy.response.headers['Access-Control-Allow-Origin'] = '*'


if __name__ == '__main__':
    cherrypy.tools.cors = cherrypy._cptools.HandlerTool(cors)
    cherrypy.config.update(globalConfig)
    cherrypy.response.timeout = 1000000000

    app = App()
    cherrypy.quickstart(app, "/")