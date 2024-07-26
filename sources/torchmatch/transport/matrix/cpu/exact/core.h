#ifndef LEMON_CORE_H
#define LEMON_CORE_H

#include <algorithm>
#include <vector>

// Disable the following warnings when compiling with MSVC:
// C4250: 'class1' : inherits 'class2::member' via dominance
// C4355: 'this' : used in base member initializer list
// C4503: 'function' : decorated name length exceeded, name was truncated
// C4800: 'type' : forcing value to bool 'true' or 'false' (performance warning)
// C4996: 'function': was declared deprecated
#ifdef _MSC_VER
#pragma warning(disable : 4250 4355 4503 4800 4996)
#endif

///\file
///\brief Graph-core utilities.
///
/// This header file contains core utilities used by the graph types in
/// this directory. It is automatically included by all graph types, therefore
/// it usually does not have to be included directly.

namespace lemon {

/// \brief Dummy type to make it easier to create invalid iterators.
///
/// Dummy type to make it easier to create invalid iterators.
/// See \ref INVALID for the usage.
struct Invalid {
  public:
    bool operator==(Invalid) { return true; }
    bool operator!=(Invalid) { return false; }
    bool operator<(Invalid) { return false; }
};

/// \brief Invalid iterators.
///
/// \ref Invalid is a global type that converts to each iterator
/// in such a way that the value of the target iterator will be invalid.
#ifdef LEMON_ONLY_TEMPLATES
const Invalid INVALID = Invalid();
#else
extern const Invalid INVALID;
#endif

/// \addtogroup gutils
/// @{

/// Create convenience typedefs for the digraph types and iterators

/// This \c \#define creates convenient type definitions for the following
/// types of \c Digraph: \c Node,  \c NodeIt, \c Arc, \c ArcIt, \c InArcIt,
///\c OutArcIt, \c BoolNodeMap, \c IntNodeMap, \c DoubleNodeMap,
///\c BoolArcMap, \c IntArcMap, \c DoubleArcMap.
///
///\note If the graph type is a dependent type, ie. the graph type depend
/// on a template parameter, then use \c TEMPLATE_DIGRAPH_TYPEDEFS()
/// macro.
#define DIGRAPH_TYPEDEFS(Digraph)                                                      \
    typedef Digraph::Node Node;                                                        \
    typedef Digraph::Arc Arc;

/// Create convenience typedefs for the digraph types and iterators

///\see DIGRAPH_TYPEDEFS
///
///\note Use this macro, if the graph type is a dependent type,
/// ie. the graph type depend on a template parameter.
#define TEMPLATE_DIGRAPH_TYPEDEFS(Digraph)                                             \
    typedef typename Digraph::Node Node;                                               \
    typedef typename Digraph::Arc Arc;

} // namespace lemon

#endif
