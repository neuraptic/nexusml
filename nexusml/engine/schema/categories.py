import copy
import json
from typing import Dict, List, Union


class Categories(object):
    """
    Class for maintain the information (categories) for all categorical attributes
    """

    def __init__(self, categories_by_element: Dict[str, List]):
        """
        Default constructor
        Args:
            categories_by_element Dict[str, List]: dict that contains the category information for each attribute
                                                The key of the dict is the element id and the value is a list
                                                with the information for each choice (see NexusML API)
        """
        # Store categories
        self.categories_by_element = copy.deepcopy(categories_by_element)

    def _has_entry(self, element_id: str) -> bool:
        return element_id in self.categories_by_element

    def has_entry(self, element_ids: Union[str, List[str]]) -> bool:
        """
        Check if the given elements ids (one or more) are present on the categories dict

        Args:
            element_ids (Union[str, List[str]]): element id or list of element ids to check

        Returns:
            bool: True if all elements are in the dict or False if not
        """
        # If we have a single id, check it
        if isinstance(element_ids, str):
            return self._has_entry(element_ids)
        else:
            # Check all individually and return True if all are present
            return all(map(self._has_entry, element_ids))

    def get_categories(self, element_id) -> List:
        """
        Get all the categories of the given element id

        Args:
            element_id (str): element id of which get the categories

        Returns:
            List with the categories of the given element
        """
        # Note: I'm assuming that 'name' attribute has the value that matches with data
        return list(map(lambda x: x['name'], self.categories_by_element[element_id]))

    @classmethod
    def create_category_from_dict(cls, d: Dict):
        """
        Class method that builds Categories object from dict

        Args:
            d (Dict): dict to use for build Categories object

        Returns:
            Categories object built with the given dictionary
        """
        return cls(categories_by_element=d)

    @classmethod
    def create_category_from_json(cls, json_file: str):
        """
        Class method that builds Categories object from JSON file

        Args:
            json_file (str): path to JSON file to use for build Categories object

        Returns:
            Categories object built with the given JSON file
        """
        # Read JSON file and create Categories object
        with open(json_file, 'r') as f:
            return cls.create_category_from_dict(json.load(f))

    @staticmethod
    def _compare(categories1, categories2) -> bool:
        """
        Function that compare two categories objects and return True if they are equal (False otherwise)

        Args:
            categories1 (Categories): first categories to compare
            categories2 (Categories): second categories to compare

        Returns:
            bool: True if two objects are equal, False otherwise
        """
        # Same number of elements
        if len(categories1.categories_by_element) != len(categories2.categories_by_element):
            return False
        # For each element
        for k, v in categories1.categories_by_element.items():
            # If not in both categories, return False
            if k not in categories2.categories_by_element:
                return False
            # If the categories are different, return False
            if v != categories2.categories_by_element[k]:
                return False
        # Every check is passed, so they are equal
        return True

    def compare(self, other_categories) -> bool:
        """
        Method that compares the current categories with the other categories, and return True if both are equal

        Args:
            other_categories (Categories): categories object which compare with

        Returns:
            bool: True if self and other categories are equeal, False otherwise
        """
        return Categories._compare(self, other_categories)
