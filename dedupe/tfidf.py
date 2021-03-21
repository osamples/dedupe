import logging
import tempfile
import sqlite3

from .index import Index
from .core import Enumerator

logger = logging.getLogger(__name__)


class TfIdfIndex(Index):
    def __init__(self):
        
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = self.temp_dir.name + '/index.db'
        self._doc_to_id = Enumerator(start=1)


        self.con = sqlite3.connect(self.db)

        # Set journal mode to WAL.
        self.con.execute('pragma journal_mode=wal')

        # consider making 'contentless'
        self.con.execute('''CREATE VIRTUAL TABLE docs USING fts5(doc, content='')''')
        self.con.execute('''create virtual table tokenize using fts3tokenize('unicode61')''')

             
    def index(self, doc):
        
        
        if doc not in self._doc_to_id:
            i = self._doc_to_id[doc]
            self.con.execute("INSERT INTO docs (rowid, doc) VALUES (?, ?)",
                             (i, doc))

    def unindex(self, doc):
        i = self._doc_to_id.pop(doc)
        raise NotImplemented

    def initSearch(self):
        #self.con.execute('''INSERT INTO docs(docs) VALUES('optimize')''')
        pass

    def search(self, doc, threshold=0):
        
        # create virtual table tok1 using fts3tokenize('porter');
        # select group_concat(token, ' OR ') from tok1 where input = 'This is a test sentence';

        query = "SELECT rowid from docs WHERE doc MATCH (select group_concat(token, ' OR ') from tokenize where input = ?) ORDER BY rank limit 3"
        results = self.con.execute(query, 
                                   (doc,))
        foo = (id for id, in results)
        return foo
