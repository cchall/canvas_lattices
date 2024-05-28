def beamline_id_list(beamlines):
    beamline_id_list = []

    for i,line in enumerate(beamlines):
        beamline_id_list.append(line['id'])
    return beamline_id_list

class Lattice:
    def __init__(self, model):
        self._beamlines = model.models.beamlines
        self._elements = model.models.elements
        
        self._beamline_id_list = beamline_id_list(self._beamlines)
        self._beamline_map = {}  # m['models']['beamlines'] index to id
        self._element_map = {} # m['models']['elements'] index to _id
        
        for i,line in enumerate(self._beamlines):
            self._beamline_map[i] = line['id']
        for i, element in enumerate(self._elements):
            self._element_map[i] = element['_id']
        
        self._inv_beamline_map = {v:k for k,v in self._beamline_map.items()}  # given beamline id given index of dict in m['models']['beamlines']
        self._inv_element_map = {v:k for k,v in self._element_map.items()}
        
    def _get_lattice_by_name(self, lattice_name):
        for i, lattice in enumerate(self._beamlines):
            if lattice.name == lattice_name:
                return i, lattice
        
    def flatten_lattice(self, lattice_name):
        beamline_map = {k: self._beamlines[v] for k, v in self._inv_beamline_map.items()}
        beamline_id_list = self._beamline_id_list
        _, lattice_dict = self._get_lattice_by_name(lattice_name)
        
        def iterate_lattice(lattice_dict):
            for element in lattice_dict['items']:  # element is id
                if element in beamline_id_list:
                    new_beamline = beamline_map[element]
                    for sub_element in iterate_lattice(new_beamline):
                        yield(sub_element)
                else:
                    # yield element
                    yield self._elements[self._inv_element_map[element]]
                    
        return iterate_lattice(lattice_dict)