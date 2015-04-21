import os

class Cache(object):
    def __init__(self, fileName, storageObject):
        self.cache = storageObject
        self.file = 'Cache/'+fileName
        self.load()
        self.saveCache = True

    def load(self):
        try:
            print 'Reading cache %s...' % self.file
            with open(self.file, 'r') as f:
                cache = f.read()
                try:
                    self.cache = eval(cache)
                except Exception, e:
                    print e
        except IOError:
            pass

    def replace(self, fileName, storageObject):
        if self.saveCache:
            oldFile = self.file
            # save new cache
            self.cache = storageObject
            self.file = fileName
            self.save()
            if self.saveCache: # self.save() may have turned saving off
                # delete old cache
                try:
                    os.remove(oldFile)
                except OSError:
                    pass

    def save(self):
        if self.saveCache:
            try:
                print 'Saving cache %s...' % self.file
                with open(self.file, 'w') as f:
                    f.write(str(self.cache))
            except IOError:
                self.saveCache = False
                print "Failed while writing cache. Cache will no longer be saved..."