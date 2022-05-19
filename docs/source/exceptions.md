# Exception handling

In general, when contributing code to Forest that utilizes exceptions, try to avoid using bare except clauses unless the desired outcome is to truly catch all possible exceptions within the try clauses and prevent them from execution. This guidance should be especially followed when attempting to log instances where an execution resulted in a well-defined error. 

For instance, below is an example of an exception usage that should be avoid:
```
# Don't do this
try:
    # Do a whole bunch of stuff
except:
    # Log that something went wrong
```

Rather it is much better practice to create custom exception classes for handing errors that typical, well-defined:
```
class FileNotFound(Exception): pass
class InvalidDataFormat(Exception): pass

try:
    # Do a whole bunch of stuff
except FileNotFound:
    # Skip this analysis
except InvalidDataFormat:
    # Try to fix data
except Exception as e:
    logging.error()
    raise e
```

Below is another example of improper exceptions usage in Forest:
```
def get_data():
    if some_condition:
        raise InvalidDataFormat
    return {"distance": distance}

# Don't do this
try:
    data = get_data()
    value = data["distence"]
    # More lines of code
except:
    value = 0

# Do this instead
try:
    data = get_data()
except InvalidDataFormat:
    data = {"distance": 0}
value = data["distence"]
# More lines of code
```
- In the above example, notice that the user has misspelled the word "distance" as "distence". As a result, an error signifying that the user is trying to access an unknown key in the dictionary should be throw. However, in the first try-except block, the user grouped the `data = get_data()` and the `value = data["distence"]` within the same try clause. As a result, if an error is thrown in either of those statements, the except clause will simply set `value = 0` without signifying to the user that they have failed to access the true "distance" value. Instead, the second try-except block highlights the appropriate usage. First, we localize they try clause to the statements that we believe need to be handled, which in this case is simply `data = get_data()`. Next, we utilize a custom except clause, where we execute the code `data = {"distance": 0}` only when the `InvalidDataFormat` is raised by the `get_data()` method. Finally, we execute `value = data["distence"]` separately, which should throw an appropriate error due to the misspelling of "distance". 
