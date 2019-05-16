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
            fp, pathname, description = imp.find_module('_ParamTot', [dirname(__file__)])
        except ImportError:
            import _ParamTot
            return _ParamTot
        if fp is not None:
            try:
                _mod = imp.load_module('_ParamTot', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _ParamTot = swig_import_helper()
    del swig_import_helper
else:
    import _ParamTot
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


class ParamTot(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ParamTot, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ParamTot, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _ParamTot.new_ParamTot(*args)
        try: self.this.append(this)
        except: self.this = this
    def H(self, *args): return _ParamTot.ParamTot_H(self, *args)
    def eta(self, *args): return _ParamTot.ParamTot_eta(self, *args)
    def rho(self, *args): return _ParamTot.ParamTot_rho(self, *args)
    def T(self, *args): return _ParamTot.ParamTot_T(self, *args)
    def K(self, *args): return _ParamTot.ParamTot_K(self, *args)
    def size(self): return _ParamTot.ParamTot_size(self)
    def xi(self): return _ParamTot.ParamTot_xi(self)
    def HTrigger(self, *args): return _ParamTot.ParamTot_HTrigger(self, *args)
    def etaTrigger(self, *args): return _ParamTot.ParamTot_etaTrigger(self, *args)
    def rhoTrigger(self, *args): return _ParamTot.ParamTot_rhoTrigger(self, *args)
    def TTrigger(self, *args): return _ParamTot.ParamTot_TTrigger(self, *args)
    __swig_destroy__ = _ParamTot.delete_ParamTot
    __del__ = lambda self : None;
ParamTot_swigregister = _ParamTot.ParamTot_swigregister
ParamTot_swigregister(ParamTot)

class ParamTotUnordered(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ParamTotUnordered, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ParamTotUnordered, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _ParamTot.new_ParamTotUnordered(*args)
        try: self.this.append(this)
        except: self.this = this
    def H(self, *args): return _ParamTot.ParamTotUnordered_H(self, *args)
    def eta(self, *args): return _ParamTot.ParamTotUnordered_eta(self, *args)
    def rho(self, *args): return _ParamTot.ParamTotUnordered_rho(self, *args)
    def T(self, *args): return _ParamTot.ParamTotUnordered_T(self, *args)
    def K(self, *args): return _ParamTot.ParamTotUnordered_K(self, *args)
    def size(self): return _ParamTot.ParamTotUnordered_size(self)
    def xi(self): return _ParamTot.ParamTotUnordered_xi(self)
    __swig_destroy__ = _ParamTot.delete_ParamTotUnordered
    __del__ = lambda self : None;
ParamTotUnordered_swigregister = _ParamTot.ParamTotUnordered_swigregister
ParamTotUnordered_swigregister(ParamTotUnordered)

class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)
    def __init__(self, *args, **kwargs): raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _ParamTot.delete_SwigPyIterator
    __del__ = lambda self : None;
    def value(self): return _ParamTot.SwigPyIterator_value(self)
    def incr(self, n = 1): return _ParamTot.SwigPyIterator_incr(self, n)
    def decr(self, n = 1): return _ParamTot.SwigPyIterator_decr(self, n)
    def distance(self, *args): return _ParamTot.SwigPyIterator_distance(self, *args)
    def equal(self, *args): return _ParamTot.SwigPyIterator_equal(self, *args)
    def copy(self): return _ParamTot.SwigPyIterator_copy(self)
    def next(self): return _ParamTot.SwigPyIterator_next(self)
    def __next__(self): return _ParamTot.SwigPyIterator___next__(self)
    def previous(self): return _ParamTot.SwigPyIterator_previous(self)
    def advance(self, *args): return _ParamTot.SwigPyIterator_advance(self, *args)
    def __eq__(self, *args): return _ParamTot.SwigPyIterator___eq__(self, *args)
    def __ne__(self, *args): return _ParamTot.SwigPyIterator___ne__(self, *args)
    def __iadd__(self, *args): return _ParamTot.SwigPyIterator___iadd__(self, *args)
    def __isub__(self, *args): return _ParamTot.SwigPyIterator___isub__(self, *args)
    def __add__(self, *args): return _ParamTot.SwigPyIterator___add__(self, *args)
    def __sub__(self, *args): return _ParamTot.SwigPyIterator___sub__(self, *args)
    def __iter__(self): return self
SwigPyIterator_swigregister = _ParamTot.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class Vector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Vector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Vector, name)
    __repr__ = _swig_repr
    def iterator(self): return _ParamTot.Vector_iterator(self)
    def __iter__(self): return self.iterator()
    def __nonzero__(self): return _ParamTot.Vector___nonzero__(self)
    def __bool__(self): return _ParamTot.Vector___bool__(self)
    def __len__(self): return _ParamTot.Vector___len__(self)
    def pop(self): return _ParamTot.Vector_pop(self)
    def __getslice__(self, *args): return _ParamTot.Vector___getslice__(self, *args)
    def __setslice__(self, *args): return _ParamTot.Vector___setslice__(self, *args)
    def __delslice__(self, *args): return _ParamTot.Vector___delslice__(self, *args)
    def __delitem__(self, *args): return _ParamTot.Vector___delitem__(self, *args)
    def __getitem__(self, *args): return _ParamTot.Vector___getitem__(self, *args)
    def __setitem__(self, *args): return _ParamTot.Vector___setitem__(self, *args)
    def append(self, *args): return _ParamTot.Vector_append(self, *args)
    def empty(self): return _ParamTot.Vector_empty(self)
    def size(self): return _ParamTot.Vector_size(self)
    def clear(self): return _ParamTot.Vector_clear(self)
    def swap(self, *args): return _ParamTot.Vector_swap(self, *args)
    def get_allocator(self): return _ParamTot.Vector_get_allocator(self)
    def begin(self): return _ParamTot.Vector_begin(self)
    def end(self): return _ParamTot.Vector_end(self)
    def rbegin(self): return _ParamTot.Vector_rbegin(self)
    def rend(self): return _ParamTot.Vector_rend(self)
    def pop_back(self): return _ParamTot.Vector_pop_back(self)
    def erase(self, *args): return _ParamTot.Vector_erase(self, *args)
    def __init__(self, *args): 
        this = _ParamTot.new_Vector(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(self, *args): return _ParamTot.Vector_push_back(self, *args)
    def front(self): return _ParamTot.Vector_front(self)
    def back(self): return _ParamTot.Vector_back(self)
    def assign(self, *args): return _ParamTot.Vector_assign(self, *args)
    def resize(self, *args): return _ParamTot.Vector_resize(self, *args)
    def insert(self, *args): return _ParamTot.Vector_insert(self, *args)
    def reserve(self, *args): return _ParamTot.Vector_reserve(self, *args)
    def capacity(self): return _ParamTot.Vector_capacity(self)
    __swig_destroy__ = _ParamTot.delete_Vector
    __del__ = lambda self : None;
Vector_swigregister = _ParamTot.Vector_swigregister
Vector_swigregister(Vector)

# This file is compatible with both classic and new-style classes.


