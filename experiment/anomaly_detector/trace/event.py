
class Event(object):
    def __init__(self, eventID):
        self.eventID

class TraceEvent(Event):
    def __init__(self,eventType):
        self.eventType=eventType

class LogEvent(Event):
    def __init__(self,tmplID):
        self.tmplID=tmplID
