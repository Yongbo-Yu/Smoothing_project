# This file was automatically generated by SWIG (http://www.swig.org).
# Version 2.0.4
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.



from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_RBergomi', [dirname(__file__)])
        except ImportError:
            import _RBergomi
            return _RBergomi
        if fp is not None:
            try:
                _mod = imp.load_module('_RBergomi', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _RBergomi = swig_import_helper()
    del swig_import_helper
else:
    import _RBergomi
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)
    def __init__(self, *args, **kwargs): raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _RBergomi.delete_SwigPyIterator
    __del__ = lambda self : None;
    def value(self): return _RBergomi.SwigPyIterator_value(self)
    def incr(self, n = 1): return _RBergomi.SwigPyIterator_incr(self, n)
    def decr(self, n = 1): return _RBergomi.SwigPyIterator_decr(self, n)
    def distance(self, *args): return _RBergomi.SwigPyIterator_distance(self, *args)
    def equal(self, *args): return _RBergomi.SwigPyIterator_equal(self, *args)
    def copy(self): return _RBergomi.SwigPyIterator_copy(self)
    def next(self): return _RBergomi.SwigPyIterator_next(self)
    def __next__(self): return _RBergomi.SwigPyIterator___next__(self)
    def previous(self): return _RBergomi.SwigPyIterator_previous(self)
    def advance(self, *args): return _RBergomi.SwigPyIterator_advance(self, *args)
    def __eq__(self, *args): return _RBergomi.SwigPyIterator___eq__(self, *args)
    def __ne__(self, *args): return _RBergomi.SwigPyIterator___ne__(self, *args)
    def __iadd__(self, *args): return _RBergomi.SwigPyIterator___iadd__(self, *args)
    def __isub__(self, *args): return _RBergomi.SwigPyIterator___isub__(self, *args)
    def __add__(self, *args): return _RBergomi.SwigPyIterator___add__(self, *args)
    def __sub__(self, *args): return _RBergomi.SwigPyIterator___sub__(self, *args)
    def __iter__(self): return self
SwigPyIterator_swigregister = _RBergomi.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class Vector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Vector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Vector, name)
    __repr__ = _swig_repr
    def iterator(self): return _RBergomi.Vector_iterator(self)
    def __iter__(self): return self.iterator()
    def __nonzero__(self): return _RBergomi.Vector___nonzero__(self)
    def __bool__(self): return _RBergomi.Vector___bool__(self)
    def __len__(self): return _RBergomi.Vector___len__(self)
    def pop(self): return _RBergomi.Vector_pop(self)
    def __getslice__(self, *args): return _RBergomi.Vector___getslice__(self, *args)
    def __setslice__(self, *args): return _RBergomi.Vector___setslice__(self, *args)
    def __delslice__(self, *args): return _RBergomi.Vector___delslice__(self, *args)
    def __delitem__(self, *args): return _RBergomi.Vector___delitem__(self, *args)
    def __getitem__(self, *args): return _RBergomi.Vector___getitem__(self, *args)
    def __setitem__(self, *args): return _RBergomi.Vector___setitem__(self, *args)
    def append(self, *args): return _RBergomi.Vector_append(self, *args)
    def empty(self): return _RBergomi.Vector_empty(self)
    def size(self): return _RBergomi.Vector_size(self)
    def clear(self): return _RBergomi.Vector_clear(self)
    def swap(self, *args): return _RBergomi.Vector_swap(self, *args)
    def get_allocator(self): return _RBergomi.Vector_get_allocator(self)
    def begin(self): return _RBergomi.Vector_begin(self)
    def end(self): return _RBergomi.Vector_end(self)
    def rbegin(self): return _RBergomi.Vector_rbegin(self)
    def rend(self): return _RBergomi.Vector_rend(self)
    def pop_back(self): return _RBergomi.Vector_pop_back(self)
    def erase(self, *args): return _RBergomi.Vector_erase(self, *args)
    def __init__(self, *args): 
        this = _RBergomi.new_Vector(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(self, *args): return _RBergomi.Vector_push_back(self, *args)
    def front(self): return _RBergomi.Vector_front(self)
    def back(self): return _RBergomi.Vector_back(self)
    def assign(self, *args): return _RBergomi.Vector_assign(self, *args)
    def resize(self, *args): return _RBergomi.Vector_resize(self, *args)
    def insert(self, *args): return _RBergomi.Vector_insert(self, *args)
    def reserve(self, *args): return _RBergomi.Vector_reserve(self, *args)
    def capacity(self): return _RBergomi.Vector_capacity(self)
    __swig_destroy__ = _RBergomi.delete_Vector
    __del__ = lambda self : None;
Vector_swigregister = _RBergomi.Vector_swigregister
Vector_swigregister(Vector)

class Result(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Result, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Result, name)
    __repr__ = _swig_repr
    __swig_setmethods__["price"] = _RBergomi.Result_price_set
    __swig_getmethods__["price"] = _RBergomi.Result_price_get
    if _newclass:price = _swig_property(_RBergomi.Result_price_get, _RBergomi.Result_price_set)
    __swig_setmethods__["iv"] = _RBergomi.Result_iv_set
    __swig_getmethods__["iv"] = _RBergomi.Result_iv_get
    if _newclass:iv = _swig_property(_RBergomi.Result_iv_get, _RBergomi.Result_iv_set)
    __swig_setmethods__["par"] = _RBergomi.Result_par_set
    __swig_getmethods__["par"] = _RBergomi.Result_par_get
    if _newclass:par = _swig_property(_RBergomi.Result_par_get, _RBergomi.Result_par_set)
    __swig_setmethods__["stat"] = _RBergomi.Result_stat_set
    __swig_getmethods__["stat"] = _RBergomi.Result_stat_get
    if _newclass:stat = _swig_property(_RBergomi.Result_stat_get, _RBergomi.Result_stat_set)
    __swig_setmethods__["N"] = _RBergomi.Result_N_set
    __swig_getmethods__["N"] = _RBergomi.Result_N_get
    if _newclass:N = _swig_property(_RBergomi.Result_N_get, _RBergomi.Result_N_set)
    __swig_setmethods__["M"] = _RBergomi.Result_M_set
    __swig_getmethods__["M"] = _RBergomi.Result_M_get
    if _newclass:M = _swig_property(_RBergomi.Result_M_get, _RBergomi.Result_M_set)
    __swig_setmethods__["numThreads"] = _RBergomi.Result_numThreads_set
    __swig_getmethods__["numThreads"] = _RBergomi.Result_numThreads_get
    if _newclass:numThreads = _swig_property(_RBergomi.Result_numThreads_get, _RBergomi.Result_numThreads_set)
    __swig_setmethods__["time"] = _RBergomi.Result_time_set
    __swig_getmethods__["time"] = _RBergomi.Result_time_get
    if _newclass:time = _swig_property(_RBergomi.Result_time_get, _RBergomi.Result_time_set)
    def __init__(self): 
        this = _RBergomi.new_Result()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _RBergomi.delete_Result
    __del__ = lambda self : None;
Result_swigregister = _RBergomi.Result_swigregister
Result_swigregister(Result)

class ResultUnordered(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ResultUnordered, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ResultUnordered, name)
    __repr__ = _swig_repr
    __swig_setmethods__["price"] = _RBergomi.ResultUnordered_price_set
    __swig_getmethods__["price"] = _RBergomi.ResultUnordered_price_get
    if _newclass:price = _swig_property(_RBergomi.ResultUnordered_price_get, _RBergomi.ResultUnordered_price_set)
    __swig_setmethods__["iv"] = _RBergomi.ResultUnordered_iv_set
    __swig_getmethods__["iv"] = _RBergomi.ResultUnordered_iv_get
    if _newclass:iv = _swig_property(_RBergomi.ResultUnordered_iv_get, _RBergomi.ResultUnordered_iv_set)
    __swig_setmethods__["par"] = _RBergomi.ResultUnordered_par_set
    __swig_getmethods__["par"] = _RBergomi.ResultUnordered_par_get
    if _newclass:par = _swig_property(_RBergomi.ResultUnordered_par_get, _RBergomi.ResultUnordered_par_set)
    __swig_setmethods__["stat"] = _RBergomi.ResultUnordered_stat_set
    __swig_getmethods__["stat"] = _RBergomi.ResultUnordered_stat_get
    if _newclass:stat = _swig_property(_RBergomi.ResultUnordered_stat_get, _RBergomi.ResultUnordered_stat_set)
    __swig_setmethods__["N"] = _RBergomi.ResultUnordered_N_set
    __swig_getmethods__["N"] = _RBergomi.ResultUnordered_N_get
    if _newclass:N = _swig_property(_RBergomi.ResultUnordered_N_get, _RBergomi.ResultUnordered_N_set)
    __swig_setmethods__["M"] = _RBergomi.ResultUnordered_M_set
    __swig_getmethods__["M"] = _RBergomi.ResultUnordered_M_get
    if _newclass:M = _swig_property(_RBergomi.ResultUnordered_M_get, _RBergomi.ResultUnordered_M_set)
    __swig_setmethods__["numThreads"] = _RBergomi.ResultUnordered_numThreads_set
    __swig_getmethods__["numThreads"] = _RBergomi.ResultUnordered_numThreads_get
    if _newclass:numThreads = _swig_property(_RBergomi.ResultUnordered_numThreads_get, _RBergomi.ResultUnordered_numThreads_set)
    __swig_setmethods__["time"] = _RBergomi.ResultUnordered_time_set
    __swig_getmethods__["time"] = _RBergomi.ResultUnordered_time_get
    if _newclass:time = _swig_property(_RBergomi.ResultUnordered_time_get, _RBergomi.ResultUnordered_time_set)
    def __init__(self): 
        this = _RBergomi.new_ResultUnordered()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _RBergomi.delete_ResultUnordered
    __del__ = lambda self : None;
ResultUnordered_swigregister = _RBergomi.ResultUnordered_swigregister
ResultUnordered_swigregister(ResultUnordered)

class RBergomiST(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, RBergomiST, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, RBergomiST, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _RBergomi.new_RBergomiST(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _RBergomi.delete_RBergomiST
    __del__ = lambda self : None;
    def ComputePayoffRT_single(self, *args): return _RBergomi.RBergomiST_ComputePayoffRT_single(self, *args)
    def getM(self): return _RBergomi.RBergomiST_getM(self)
    def setM(self, *args): return _RBergomi.RBergomiST_setM(self, *args)
    def getN(self): return _RBergomi.RBergomiST_getN(self)
    def setN(self, *args): return _RBergomi.RBergomiST_setN(self, *args)
    def getXi(self): return _RBergomi.RBergomiST_getXi(self)
RBergomiST_swigregister = _RBergomi.RBergomiST_swigregister
RBergomiST_swigregister(RBergomiST)

# This file is compatible with both classic and new-style classes.


