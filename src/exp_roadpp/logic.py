
from abc import ABC, abstractmethod
import torch


class DataType(object):
    """Data type in first-order logic.

    A class of data types in first-order logic.

    Args:
        name (str): The name of the data type.

    Attrs:
        name (str): The name of the data type.
    """

    def __init__(self, data):
        self.data = data.split(',')
        if len(self.data) != 2:
            raise ValueError
        self.name = self.data[0]
        self.sign = self.data[1]

    def __eq__(self, other):
        if type(other) == str:
            return self.name == other
        else:
            return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())


class ModeTerm(object):
    """Terms for mode declarations. It has mode (+, -, #) and data types.
    """

    def __init__(self, mode, dtype):
        self.mode = mode
        assert mode in ['+', '-', '#'], "Invalid mode declaration."
        self.dtype = dtype

    def __str__(self):
        return self.mode + self.dtype.name

    def __repr__(self):
        return self.__str__()


def flatten(x): return [z for y in x for z in (
    flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]


class Term(ABC):
    """Terms in first-order logic.

    An abstract class of terms in first-oder logic.

    Attributes:
        name (str): Name of the term.
        dtype (datatype): Data type of the term.
    """

    @abstractmethod
    def __repr__(self, level=0):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def all_vars(self):
        pass

    @abstractmethod
    def all_consts(self):
        pass

    @abstractmethod
    def all_funcs(self):
        pass

    @abstractmethod
    def max_depth(self):
        pass

    @abstractmethod
    def min_depth(self):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def is_var(self):
        pass



class Var(Term):
    """Variables in first-order logic.

    A class of variable in first-oder logic.

    Attributes:
        name (str): Name of the variable.
    """

    def __init__(self, name, var_type):
        tokes = name.split("_")
        self.var_type = var_type
        if len(tokes) == 1:
            self.name = name
            self.id = None
        elif len(tokes) == 2:
            self.name = name
            self.id = int(tokes[1])
        self.name_family = name

    def __repr__(self, level=0):
        # ret = "\t"*level+repr(self.name)+"\n"
        ret = self.name
        return ret

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return type(other) == Var and self.name == other.name

    def __hash__(self):
        return hash(self.__str__())

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    def head(self):
        return self

    def subs(self, target_var, const):
        if self.name == target_var.name:
            return const
        else:
            return self

    def to_list(self):
        return [self]

    def get_ith_term(self, i):
        assert i == 0, 'Invalid ith term for constant ' + str(self)
        return self

    def all_vars(self):
        return [self]

    def all_consts(self):
        return []

    def all_funcs(self):
        return []

    def max_depth(self):
        return 0

    def min_depth(self):
        return 0

    def size(self):
        return 1

    def is_var(self):
        return 1



class Atom(object):
    """Atoms in first-order logic.

    A class of atoms: p(t1, ..., tn)

    Attributes:
        pred (Predicate): A predicate of the atom.
        terms (List[Term]): The terms for the atoms.
    """

    def __init__(self, pred, terms):
        # if pred.arity != len(terms):
        #     print(f"pred:{pred}, terms:{terms}, arity:{pred.arity}")
        #     raise ValueError(f'Invalid arguments for predicate symbol {pred.name}')

        self.pred = pred
        self.terms = terms
        self.neg_state = False

    def __eq__(self, other):
        if other == None:
            return False
        if self.pred == other.pred:
            for i in range(len(self.terms)):
                if not self.terms[i] == other.terms[i]:
                    return False
            return True
        else:
            return False

    def __str__(self):
        s = self.pred.name + '('
        for arg in self.terms:
            s += arg.__str__() + ','
        s = s[0:-1]
        s += ')'
        return s

    def __hash__(self):
        return hash(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        """comparison < """
        return self.__str__() < other.__str__()

    def __gt__(self, other):
        """comparison > """
        return self.__str__() > other.__str__()

    def subs(self, target_var, const):
        self.terms = [term.subs(target_var, const) for term in self.terms]

    def neg(self):
        self.neg_state = not self.neg_state

    def all_vars(self):
        var_list = []
        for term in self.terms:
            # var_list.append(term.all_vars())
            var_list += term.all_vars()
        return var_list

    def all_consts(self):
        const_list = []
        for term in self.terms:
            const_list += term.all_consts()
        return const_list

    def all_funcs(self):
        func_list = []
        for term in self.terms:
            func_list += term.all_funcs()
        return func_list

    def max_depth(self):
        return max([term.max_depth() for term in self.terms])

    def min_depth(self):
        return min([term.min_depth() for term in self.terms])

    def size(self):
        size = 0
        for term in self.terms:
            size += term.size()
        return size

    def get_terms_by_dtype(self, dtype):
        """Return terms that have type of dtype.
        Returns: (list(Term))
        """
        result = []
        for i, term in enumerate(self.terms):
            if self.pred.dtypes[i] == dtype:
                # print( self.pred.dtypes[i], dtype,  self.pred.dtypes[i] == dtype)
                result.append(term)

        return result


    
class Clause(object):
    """Clauses in first-oder logic.

    A class of clauses in first-order logic: A :- B1, ..., Bn.

    Attributes:
        head (Atom): The head atom.
        body (List[Atom]): The atoms for the body.
    """

    def __init__(self, head, body):
        self.head = head
        self.body = sorted(body)
        # self.body = body
        # print(self)
        ###self._rename()
        # print(self)

    def __str__(self):

        head_str = self.head.__str__()
        body_str = ""
        for bi in self.body:
            body_str += bi.__str__()
            body_str += ','
        body_str = body_str[0:-1]
        body_str += '.'
        return head_str + ':-' + body_str

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        # return self._id_str() == other._id_str()
        return self.head == other.head and set(self.body) == set(other.body)
        # return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    def __gt__(self, other):
        return self.__str__() > other.__str__()

    def _rename(self):
        """Renaming variables to compute the equality.
        e.g. p(O1,O2):-. == p(O2,O3):-. == p(__X1__,__X2__):-.
        """
        atoms = [self.head] + self.body
        vars = self.all_vars()
        id_vars = [Var("_X" + str(i) + "_") for i in range(len(vars))]

        head_terms = []
        for term in self.head.terms:
            if term.is_var():
                head_terms.append(id_vars[vars.index(term)])
            else:
                head_terms.append(term)
        head_ = Atom(self.head.pred, head_terms)

        body_ = []
        for bi in self.body:
            bi_terms = []
            for term in bi.terms:
                if term.is_var():
                    bi_terms.append(id_vars[vars.index(term)])
                else:
                    bi_terms.append(term)
            bi_atom = Atom(bi.pred, bi_terms)
            body_.append(bi_atom)
        self.head = head_
        self.body = sorted(body_)

    def _id_str(self):
        """Renaming variables to compute the equality.
        e.g. p(O1,O2):-. == p(O2,O3):-. == p(__X1__,__X2__):-.
        """
        atoms = [self.head] + self.body
        vars = list(set(self.all_vars()))
        id_vars = [Var("__X" + str(i) + "__") for i in range(len(vars))]

        head_terms = []
        for term in self.head.terms:
            if term.is_var():
                head_terms.append(id_vars[vars.index(term)])
            else:
                head_terms.append(term)
        head_ = Atom(self.head.pred, head_terms)

        body_ = []
        for bi in self.body:
            bi_terms = []
            for term in bi.terms:
                if term.is_var():
                    bi_terms.append(id_vars[vars.index(term)])
                else:
                    bi_terms.append(term)
            bi_atom = Atom(bi.pred, bi_terms)
            body_.append(bi_atom)
        # to str
        head_str = head_.__str__()
        body_str = ""
        for bi in body_:
            body_str += bi.__str__()
            body_str += ','
        body_str = body_str[0:-1]
        body_str += '.'
        return head_str + ':-' + body_str

    def is_tautology(self):
        return len(self.body) == 1 and self.body[0] == self.head

    def is_duplicate(self):
        if len(self.body) >= 2:
            es = self.body
            return es == [es[0]] * len(es) if es else False
        return False

    def subs(self, target_var, const):
        if type(self.head) == Atom:
            self.head.subs(target_var, const)
        for bi in self.body:
            bi.subs(target_var, const)

    def all_vars(self):
        var_list = []
        var_list += self.head.all_vars()
        for bi in self.body:
            var_list += bi.all_vars()
        var_list = flatten(var_list)
        # remove duplication
        result = []
        for v in var_list:
            if not v in result:
                result.append(v)
        return result

    def all_vars_by_dtype(self, dtype):
        """Get all variables in the clause that has given data type.
        Returns: list(Var)
        """
        atoms = [self.head] + self.body
        result = []
        for atom in atoms:
            terms = atom.get_terms_by_dtype(dtype)
            vars = [t for t in terms if t.is_var()]
            result.extend(vars)
        return sorted(list(set(result)))

    def count_by_predicate(self, pred):
        atoms = [self.head] + self.body
        n = 0
        for atom in atoms:
            if pred == atom.pred:
                n += 1
        return n

    def all_consts(self):
        const_list = []
        const_list += self.head.all_consts()
        for bi in self.body:
            const_list += bi.all_consts()
        const_list = flatten(const_list)
        return const_list

    def all_funcs(self):
        func_list = []
        func_list += self.head.all_funcs()
        for bi in self.body:
            func_list += bi.all_funcs()
        func_list = flatten(func_list)
        return func_list

    def max_depth(self):
        depth_list = [self.head.max_depth()]
        for b in self.body:
            depth_list.append(b.max_depth())
        return max(depth_list)

    def min_depth(self):
        depth_list = [self.head.min_depth()]
        for b in self.body:
            depth_list.append(b.min_depth())
        return min(depth_list)

    def size(self):
        size = self.head.size()
        for bi in self.body:
            size += bi.size()
        return size

class Predicate():
    """Predicats in first-order logic.

    A class of predicates in first-order logic.

    Attributes:
        name (str): A name of the predicate.
        arity (int): The arity of the predicate.
        dtypes (List[DataTypes]): The data types of the arguments for the predicate.
    """

    def __init__(self, name, arity, dtypes, field_names=None): 
        
        self.name = name
        self.arity = arity
        self.dtypes = dtypes  # mode = List[dtype]
        self.field_names = tuple(field_names) if field_names else None
        if self.field_names and len(self.field_names) != self.arity:
            raise ValueError(f"Field names length {len(self.field_names)} does not match arity {self.arity} for predicate {self.name}.")

    def make_atom(self, **kwargs):
        if not self.field_names:
            raise ValueError(f"Predicate {self.name} does not have field names defined.")
        missing = set(self.field_names) - set(kwargs)
        extra = set(kwargs) - set(self.field_names)
        if missing:
            raise ValueError(f"Missing fields for predicate {self.name}: {missing}")
        if extra:
            raise ValueError(f"Extra fields for predicate {self.name}: {extra}")
        terms = [kwargs[field] for field in self.field_names]
        return Atom(self, terms)

    def __str__(self):
        # return self.name
        return self.name + '/' + str(self.arity) + '/' + str(self.dtypes)

    def __hash__(self):
        return hash(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if type(other) == Predicate:
            return self.name == other.name
        else:
            return False

    def __lt__(self, other):
        return self.__str__() < other.__str__()


class Const(Term):
    """Constants in first-order logic.

    A class of constants in first-oder logic.

    Attributes:
        name (str): Name of the term.
        dtype (datatype): Data type of the term.
    """

    def __init__(self, name, dtype=None, values=None):
        self.name = name
        self.dtype = dtype
        self.values = values
        if 'phi' in name:
            total = int(name.split('of')[-1])
            index = int(name.split('of')[0].split("phi")[-1])
            section = int(360 / total)
            values = torch.arange((index - 1) * section, (index) * section, section, dtype=torch.float)[:1]
            self.values = values
        elif "rho" in name:
            total = int(name.split('of')[-1])
            index = int(name.split('of')[0].split("rho")[-1])
            section = 1 / total
            values = torch.arange((index - 1) * section, (index) * section, step=section, dtype=torch.float)[:1]
            self.values = values

    def __repr__(self, level=0):
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return type(other) == Const and self.name == other.name

    def __hash__(self):
        return hash(self.__str__())

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    def head(self):
        return self

    def subs(self, target_var, const):
        return self

    def to_list(self):
        return [self]

    def get_ith_term(self, i):
        assert i == 0, 'Invalid ith term for constant ' + str(self)
        return self

    def all_vars(self):
        return []

    def all_consts(self):
        return [self]

    def all_funcs(self):
        return []

    def max_depth(self):
        return 0

    def min_depth(self):
        return 0

    def size(self):
        return 1

    def is_var(self):
        return 0

p_ = Predicate('.', 1, [DataType('spec,?')])
false = Atom(p_, [Const('__F__', dtype=DataType('spec,?'))])
true = Atom(p_, [Const('__T__', dtype=DataType('spec,?'))])