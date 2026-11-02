"""Expected type error: confirm the installed package's hints are consumed."""

from quatops import scalar

scalar.norm("not a quaternion")
