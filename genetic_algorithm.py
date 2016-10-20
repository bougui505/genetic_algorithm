#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2016-09-28 13:28:22 (UTC+0200)

import numpy
import scipy.optimize

class GA(object):
    """
    Genetic algorithm
    """
    def __init__(self, components, target, size=10, n_ensemble=1000,
                 crossing_freq=.8, mutation_freq=1.):
        """
        • n_ensemble: number of ensembles to generate
        • size: size of the generated ensembles
        """
        self.crossing_freq = crossing_freq
        self.mutation_freq = mutation_freq
        self.components = components
        self.target = target
        self.size = size
        self.n_ensemble = n_ensemble
        self.n_components = self.components.shape[0]
        self.ensemble = None
        self.model = None
        self.population = None
        self.chi2 = None
        self.scale = None
        self.offset = None
        self.weights = None
        self.component_ids = None

    def get_chi2(self, args):
        """
        Compute the Chi2
        """
        scale, offset = args
        chi2 = ((scale * (self.model + offset) - self.target[:, 1])**2 /\
                self.target[:, 2]**2).mean()
        return chi2

    def weight_ensemble(self, weights):
        """
        compute the weights that minimize Chi2
        """
        self.model = (self.ensemble * weights[:, None]).sum(axis=0)
        chi2 = self.get_chi2((1., 0.))
        return chi2

    def generate_ensemble(self):
        """
        generate a random ensemble
        """
        component_id = numpy.random.choice(self.n_components, self.size,
                                           replace=False)
        ensemble = self.components[component_id]
        self.ensemble = ensemble
        return component_id

    def generate_parents(self):
        """
        generate the initial population of parents
        """
        parents = []
        self.chi2 = []
        self.scale = []
        self.offset = []
        self.weights = []
        self.component_ids = []
        for _ in range(self.n_ensemble):
            self.component_ids.append(self.generate_ensemble())
            parents.append(self.ensemble)
            chi2, scale, offset, weights = self.minimize_chi2()
            self.chi2.append(chi2)
            self.scale.append(scale)
            self.offset.append(offset)
            self.weights.append(weights)
        self.population = numpy.asarray(parents)

    def generate_offspring(self):
        """
        generate offspring from parents
        """
        offspring = numpy.copy(self.population)
        # Mutations
        mutant_index_list = numpy.random.choice(self.n_ensemble,
                                                size=int(self.n_ensemble *\
                                                    self.mutation_freq))
        offspring_component_ids = numpy.copy(self.component_ids)
        for mutant_index in mutant_index_list:
            gene_index = numpy.random.choice(self.size)
            new_gene_index = numpy.random.choice(self.n_components)
            offspring[mutant_index][gene_index] =\
                                                 self.components[new_gene_index]
            offspring_component_ids[mutant_index][gene_index] = new_gene_index
        # Crossing over
        cross_index_list = numpy.random.choice(self.n_ensemble,
                                               size=int(self.n_ensemble *\
                                                    self.crossing_freq))
        for cross_index_1 in cross_index_list:
            cross_index_2 = numpy.random.choice(cross_index_list)
            gene_indices = numpy.random.choice(self.size,
                                               size=\
                                               numpy.random.choice(\
                                            range(1, self.size)), replace=False)
            part_1 = numpy.copy(offspring[cross_index_1][gene_indices])
            part_2 = numpy.copy(offspring[cross_index_2][gene_indices])
            ids_1 = numpy.copy(offspring_component_ids[cross_index_1]\
                                                                 [gene_indices])
            ids_2 = numpy.copy(offspring_component_ids[cross_index_2]\
                                                                 [gene_indices])
            offspring[cross_index_1][gene_indices] = part_2
            offspring[cross_index_2][gene_indices] = part_1
            offspring_component_ids[cross_index_1][gene_indices] = ids_2
            offspring_component_ids[cross_index_2][gene_indices] = ids_1
        self.population = numpy.r_[self.population, offspring]
        self.component_ids.extend(offspring_component_ids)
        self.component_ids = numpy.asarray(self.component_ids)
        for self.ensemble in offspring:
            chi2, scale, offset, weights = self.minimize_chi2()
            self.chi2.append(chi2)
            self.scale.append(scale)
            self.offset.append(offset)
            self.weights.append(weights)
        self.chi2 = numpy.asarray(self.chi2)
        self.scale = numpy.asarray(self.scale)
        self.offset = numpy.asarray(self.offset)
        self.weights = numpy.asarray(self.weights)

    def selection(self):
        """
        Select the self.n_ensemble best fitting individues
        """
        sorter = numpy.argsort(self.chi2)
        self.chi2 = list(self.chi2[sorter][:self.n_ensemble])
        self.scale = list(self.scale[sorter][:self.n_ensemble])
        self.offset = list(self.offset[sorter][:self.n_ensemble])
        self.weights = list(self.weights[sorter][:self.n_ensemble])
        self.population = self.population[sorter][:self.n_ensemble]
        self.component_ids = list(self.component_ids[sorter][:self.n_ensemble])

    def minimize_chi2(self):
        """
        Minimize the chi2 by finding the best weights then the best scaling and
        offset.
        """
        cons = ({'type': 'eq', 'fun': lambda x: sum(x) == 1})
        size = self.size
        weights = numpy.ones(size)*1/float(size)
        res = scipy.optimize.minimize(self.weight_ensemble, weights,
                                      constraints=cons,
                                      bounds=[(0, 1), ] * size)
        weights = res.x
        self.model = (self.ensemble * weights[:, None]).sum(axis=0)
        res = scipy.optimize.minimize(self.get_chi2, [1., 0.])
        scale, offset = res.x
        chi2 = self.get_chi2((scale, offset))
        return chi2, scale, offset, weights