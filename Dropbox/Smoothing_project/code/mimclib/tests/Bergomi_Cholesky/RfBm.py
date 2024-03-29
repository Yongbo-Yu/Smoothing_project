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
            fp, pathname, description = imp.find_module('_RfBm', [dirname(__file__)])
        except ImportError:
            import _RfBm
            return _RfBm
        if fp is not None:
            try:
                _mod = imp.load_module('_RfBm', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _RfBm = swig_import_helper()
    del swig_import_helper
else:
    import _RfBm
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
    __swig_destroy__ = _RfBm.delete_SwigPyIterator
    __del__ = lambda self : None;
    def value(self): return _RfBm.SwigPyIterator_value(self)
    def incr(self, n = 1): return _RfBm.SwigPyIterator_incr(self, n)
    def decr(self, n = 1): return _RfBm.SwigPyIterator_decr(self, n)
    def distance(self, *args): return _RfBm.SwigPyIterator_distance(self, *args)
    def equal(self, *args): return _RfBm.SwigPyIterator_equal(self, *args)
    def copy(self): return _RfBm.SwigPyIterator_copy(self)
    def next(self): return _RfBm.SwigPyIterator_next(self)
    def __next__(self): return _RfBm.SwigPyIterator___next__(self)
    def previous(self): return _RfBm.SwigPyIterator_previous(self)
    def advance(self, *args): return _RfBm.SwigPyIterator_advance(self, *args)
    def __eq__(self, *args): return _RfBm.SwigPyIterator___eq__(self, *args)
    def __ne__(self, *args): return _RfBm.SwigPyIterator___ne__(self, *args)
    def __iadd__(self, *args): return _RfBm.SwigPyIterator___iadd__(self, *args)
    def __isub__(self, *args): return _RfBm.SwigPyIterator___isub__(self, *args)
    def __add__(self, *args): return _RfBm.SwigPyIterator___add__(self, *args)
    def __sub__(self, *args): return _RfBm.SwigPyIterator___sub__(self, *args)
    def __iter__(self): return self
SwigPyIterator_swigregister = _RfBm.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class Vector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Vector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Vector, name)
    __repr__ = _swig_repr
    def iterator(self): return _RfBm.Vector_iterator(self)
    def __iter__(self): return self.iterator()
    def __nonzero__(self): return _RfBm.Vector___nonzero__(self)
    def __bool__(self): return _RfBm.Vector___bool__(self)
    def __len__(self): return _RfBm.Vector___len__(self)
    def pop(self): return _RfBm.Vector_pop(self)
    def __getslice__(self, *args): return _RfBm.Vector___getslice__(self, *args)
    def __setslice__(self, *args): return _RfBm.Vector___setslice__(self, *args)
    def __delslice__(self, *args): return _RfBm.Vector___delslice__(self, *args)
    def __delitem__(self, *args): return _RfBm.Vector___delitem__(self, *args)
    def __getitem__(self, *args): return _RfBm.Vector___getitem__(self, *args)
    def __setitem__(self, *args): return _RfBm.Vector___setitem__(self, *args)
    def append(self, *args): return _RfBm.Vector_append(self, *args)
    def empty(self): return _RfBm.Vector_empty(self)
    def size(self): return _RfBm.Vector_size(self)
    def clear(self): return _RfBm.Vector_clear(self)
    def swap(self, *args): return _RfBm.Vector_swap(self, *args)
    def get_allocator(self): return _RfBm.Vector_get_allocator(self)
    def begin(self): return _RfBm.Vector_begin(self)
    def end(self): return _RfBm.Vector_end(self)
    def rbegin(self): return _RfBm.Vector_rbegin(self)
    def rend(self): return _RfBm.Vector_rend(self)
    def pop_back(self): return _RfBm.Vector_pop_back(self)
    def erase(self, *args): return _RfBm.Vector_erase(self, *args)
    def __init__(self, *args): 
        this = _RfBm.new_Vector(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(self, *args): return _RfBm.Vector_push_back(self, *args)
    def front(self): return _RfBm.Vector_front(self)
    def back(self): return _RfBm.Vector_back(self)
    def assign(self, *args): return _RfBm.Vector_assign(self, *args)
    def resize(self, *args): return _RfBm.Vector_resize(self, *args)
    def insert(self, *args): return _RfBm.Vector_insert(self, *args)
    def reserve(self, *args): return _RfBm.Vector_reserve(self, *args)
    def capacity(self): return _RfBm.Vector_capacity(self)
    __swig_destroy__ = _RfBm.delete_Vector
    __del__ = lambda self : None;
Vector_swigregister = _RfBm.Vector_swigregister
Vector_swigregister(Vector)

class VVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, VVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, VVector, name)
    __repr__ = _swig_repr
    def iterator(self): return _RfBm.VVector_iterator(self)
    def __iter__(self): return self.iterator()
    def __nonzero__(self): return _RfBm.VVector___nonzero__(self)
    def __bool__(self): return _RfBm.VVector___bool__(self)
    def __len__(self): return _RfBm.VVector___len__(self)
    def pop(self): return _RfBm.VVector_pop(self)
    def __getslice__(self, *args): return _RfBm.VVector___getslice__(self, *args)
    def __setslice__(self, *args): return _RfBm.VVector___setslice__(self, *args)
    def __delslice__(self, *args): return _RfBm.VVector___delslice__(self, *args)
    def __delitem__(self, *args): return _RfBm.VVector___delitem__(self, *args)
    def __getitem__(self, *args): return _RfBm.VVector___getitem__(self, *args)
    def __setitem__(self, *args): return _RfBm.VVector___setitem__(self, *args)
    def append(self, *args): return _RfBm.VVector_append(self, *args)
    def empty(self): return _RfBm.VVector_empty(self)
    def size(self): return _RfBm.VVector_size(self)
    def clear(self): return _RfBm.VVector_clear(self)
    def swap(self, *args): return _RfBm.VVector_swap(self, *args)
    def get_allocator(self): return _RfBm.VVector_get_allocator(self)
    def begin(self): return _RfBm.VVector_begin(self)
    def end(self): return _RfBm.VVector_end(self)
    def rbegin(self): return _RfBm.VVector_rbegin(self)
    def rend(self): return _RfBm.VVector_rend(self)
    def pop_back(self): return _RfBm.VVector_pop_back(self)
    def erase(self, *args): return _RfBm.VVector_erase(self, *args)
    def __init__(self, *args): 
        this = _RfBm.new_VVector(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(self, *args): return _RfBm.VVector_push_back(self, *args)
    def front(self): return _RfBm.VVector_front(self)
    def back(self): return _RfBm.VVector_back(self)
    def assign(self, *args): return _RfBm.VVector_assign(self, *args)
    def resize(self, *args): return _RfBm.VVector_resize(self, *args)
    def insert(self, *args): return _RfBm.VVector_insert(self, *args)
    def reserve(self, *args): return _RfBm.VVector_reserve(self, *args)
    def capacity(self): return _RfBm.VVector_capacity(self)
    __swig_destroy__ = _RfBm.delete_VVector
    __del__ = lambda self : None;
VVector_swigregister = _RfBm.VVector_swigregister
VVector_swigregister(VVector)

class RfBm(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, RfBm, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, RfBm, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _RfBm.new_RfBm(*args)
        try: self.this.append(this)
        except: self.this = this
    def generate(self, *args): return _RfBm.RfBm_generate(self, *args)
    def __call__(self, *args): return _RfBm.RfBm___call__(self, *args)
    def GetL(self): return _RfBm.RfBm_GetL(self)
    def GetA(self): return _RfBm.RfBm_GetA(self)
    __swig_destroy__ = _RfBm.delete_RfBm
    __del__ = lambda self : None;
RfBm_swigregister = _RfBm.RfBm_swigregister
RfBm_swigregister(RfBm)

# This file is compatible with both classic and new-style classes.


